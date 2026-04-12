from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
import cv2
import uuid
import os
import shutil
import hashlib
import numpy as np
import urllib.request
import urllib.error
import json
import logging
from sklearn.metrics.pairwise import cosine_distances
from dotenv import load_dotenv

load_dotenv()

from repository import insert_faces, FaceRow
from repository import upsert_photo_by_sha1
from repository import (
    fetch_all_person_centroids,
    fetch_face_ids_by_photo,
    set_face_person_id,
    fetch_embeddings_by_person_id,
    update_person_centroid,
    insert_person,
    get_person_count,
    count_distinct_photos_for_person,
)

from person_vs_scenery.haar import HaarDetector
from person_vs_scenery.hog import HOGDetector
from person_vs_scenery.yolov8 import YOLOv8Detector

from face_detectors.mediapipe_face import MediaPipeFaceDetector
from face_detectors.insightface_face import InsightFaceDetector

logger = logging.getLogger(__name__)

app = FastAPI()

SPRING_API_BASE = os.getenv("SPRING_API_BASE", "http://localhost:8081/bitirme")

face_scenery_detectors = {
    "haar": HaarDetector(),
    "hog": HOGDetector(),
    "yolov8": YOLOv8Detector(),
}

face_detectors = {
    "mediapipe-face": MediaPipeFaceDetector(),
    "insightface": InsightFaceDetector(ctx_id=-1),  # CPU
}

PERSON_MATCH_THRESHOLD = 0.35  # cosine distance; matches DBSCAN eps


def match_embedding_to_person(embedding, persons):
    """
    Compare a face embedding against all person centroids.
    Returns (person_id, display_name, distance) or (None, None, None).
    """
    if not persons:
        return None, None, None

    centroids = np.stack([p[2] for p in persons])
    dists = cosine_distances(embedding.reshape(1, -1), centroids)[0]
    best_idx = int(np.argmin(dists))
    best_dist = float(dists[best_idx])

    if best_dist < PERSON_MATCH_THRESHOLD:
        pid, name, _ = persons[best_idx]
        return pid, name, best_dist

    return None, None, None


def _save_upload_to_temp(file: UploadFile) -> str:
    # Preserve extension if possible
    _, ext = os.path.splitext(file.filename or "")
    ext = ext if ext else ".jpg"
    temp_filename = f"temp_{uuid.uuid4().hex}{ext}"

    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return temp_filename


@app.post("/person_vs_scenery")
async def analyze_image(
    file: UploadFile = File(...),
    model: str = Query("haar"),
):
    detector = face_scenery_detectors.get(model)
    if detector is None:
        raise HTTPException(status_code=400, detail="Unknown model")

    temp_filename = _save_upload_to_temp(file)
    try:
        image = cv2.imread(temp_filename)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not read image file")

        result = detector.detect(image)
        image_type = "PEOPLE" if result["people"] > 0 else "SCENERY"

        return {
            "model": detector.name,
            "faces_or_people": result["people"],
            "confidence": result["confidence"],
            "type": image_type,
        }
    finally:
        try:
            os.remove(temp_filename)
        except Exception:
            pass


@app.post("/detect-faces")
async def detect_faces(
    file: UploadFile = File(...),
    model: str = Query("mediapipe-face"),
):
    detector = face_detectors.get(model)
    if detector is None:
        raise HTTPException(status_code=400, detail="Unknown face detector")

    temp_filename = _save_upload_to_temp(file)
    try:
        image = cv2.imread(temp_filename)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not read image file")

        result = detector.detect(image)

        # JSON-safe extraction
        faces_list = result["faces"] if isinstance(result.get("faces"), list) else []
        faces_count = result.get("faces_count", result.get("faces", 0))

        return {
            "model": detector.name,
            "faces": int(faces_count),
            "confidence": float(result.get("confidence", 0.0)),
            "boxes": [f["bbox"] for f in faces_list],
            "scores": [float(f["det_score"]) for f in faces_list],
        }
    finally:
        try:
            os.remove(temp_filename)
        except Exception:
            pass

@app.post("/ingest-faces")
async def ingest_faces(
    file: UploadFile = File(...),
    model: str = Query("insightface")
):
    detector = face_detectors.get(model)
    if detector is None:
        raise HTTPException(status_code=400, detail="Unknown face detector")

    # Enforce embedding-capable detector
    if detector.name != "insightface":
        raise HTTPException(status_code=400, detail="ingest-faces requires model=insightface (embeddings needed)")

    # 1) Read bytes once
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    # 2) SHA1 identity (same bytes => same image row)
    sha1 = hashlib.sha1(data).hexdigest()

    # 3) Decode to OpenCV image (BGR)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    h, w = img.shape[:2]

    # 4) Use original filename
    original_filename = file.filename or f"sha1/{sha1}"

    # 5) Upsert photo row by sha1
    photo_id = upsert_photo_by_sha1(
        sha1=sha1,
        original_filename=original_filename,
        width=w,
        height=h,
    )

    # 6) Detect faces + embeddings
    det = detector.detect(img)
    faces = det.get("faces", []) if isinstance(det.get("faces"), list) else []

    # 7) Stable ordering => stable face_index across runs
    faces = sorted(faces, key=lambda f: (f["bbox"][1], f["bbox"][0]))  # (y1, x1)

    face_rows = []
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = f["bbox"]
        face_rows.append(
            FaceRow(
                face_index=i,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                det_score=float(f["det_score"]),
                embedding=f["embedding"],  # np.float32 (512,)
            )
        )

    inserted = insert_faces(photo_id=photo_id, faces=face_rows, replace_existing=True)

    # 8) Online person matching
    person_assignments = []
    if face_rows:
        persons = fetch_all_person_centroids()
        face_id_map = fetch_face_ids_by_photo(photo_id)

        for (face_id, face_index), face_row in zip(face_id_map, face_rows):
            pid, pname, dist = match_embedding_to_person(face_row.embedding, persons)

            if pid is not None:
                set_face_person_id(face_id, pid)

                # Recompute centroid with newly assigned face
                all_embs = fetch_embeddings_by_person_id(pid)
                new_centroid = all_embs.mean(axis=0)
                update_person_centroid(pid, new_centroid)

                # Update local cache for subsequent faces in this image
                for i, (p_id, p_name, _) in enumerate(persons):
                    if p_id == pid:
                        persons[i] = (p_id, p_name, new_centroid)
                        break

            person_assignments.append({
                "face_index": face_index,
                "person_id": pid,
                "display_name": pname,
                "distance": round(dist, 4) if dist is not None else None,
            })

    return {
        "sha1": sha1,
        "photo_id": photo_id,
        "original_filename": original_filename,
        "faces_detected": len(face_rows),
        "db_rows_affected": inserted,
        "person_assignments": person_assignments,
    }

class AnalyzePhotoRequest(BaseModel):
    photo_id: int          # Spring Boot photo ID — used to post tags back
    file_path: str         # Absolute OS path to the saved image file


def _post_face_tag_to_spring(photo_id: int, person_name: str) -> bool:
    """
    POST /photos/{photo_id}/tags to the Spring Boot backend with
    { "name": person_name, "tagType": "FACE", "source": "SYSTEM" }
    Returns True on success.
    """
    url = f"{SPRING_API_BASE}/photos/{photo_id}/tags"
    payload = json.dumps({
        "name": person_name,
        "tagType": "FACE",
        "source": "SYSTEM",
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.getcode()
            if status in (200, 201):
                logger.info("Posted FACE tag '%s' to photo %d", person_name, photo_id)
                return True
            logger.warning("Unexpected status %d posting tag to photo %d", status, photo_id)
            return False
    except urllib.error.URLError as exc:
        logger.error("Failed to post tag to Spring Boot: %s", exc)
        return False


@app.post("/analyze-photo")
async def analyze_photo(req: AnalyzePhotoRequest):
    """
    Full auto-tagging pipeline triggered by Spring Boot after a photo upload.

    1. Read image from disk at req.file_path
    2. Classify as PEOPLE or SCENERY (YOLOv8)
    3. If SCENERY → return immediately (no tags added)
    4. If PEOPLE  → detect faces (InsightFace), match to known persons,
                    create new 'Person N' entries for unknowns,
                    POST a FACE tag for each unique person back to Spring Boot
    """
    img = cv2.imread(req.file_path)
    if img is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read image at path: {req.file_path}",
        )

    # ── 1. Person-vs-scenery classification ────────────────────────────────
    yolo_detector = face_scenery_detectors["yolov8"]
    scene_result  = yolo_detector.detect(img)
    image_type    = "PEOPLE" if scene_result["people"] > 0 else "SCENERY"

    if image_type == "SCENERY":
        return {
            "photo_id": req.photo_id,
            "type": "SCENERY",
            "action": "skipped",
            "tags_added": [],
        }

    # ── 2. Face detection + embedding (InsightFace) ────────────────────────
    insight_detector = face_detectors["insightface"]
    det = insight_detector.detect(img)
    faces = det.get("faces", []) if isinstance(det.get("faces"), list) else []

    if not faces:
        # Detected as people-photo by YOLO but InsightFace found no faces
        return {
            "photo_id": req.photo_id,
            "type": "PEOPLE",
            "action": "no_faces_detected",
            "tags_added": [],
        }

    # Stable ordering: top-to-bottom, left-to-right
    faces = sorted(faces, key=lambda f: (f["bbox"][1], f["bbox"][0]))

    # ── 2b. Persist face rows (bbox + embedding) to the faces table ────────
    face_rows = []
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = f["bbox"]
        face_rows.append(
            FaceRow(
                face_index=i,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                det_score=float(f["det_score"]),
                embedding=f["embedding"],
            )
        )
    insert_faces(photo_id=req.photo_id, faces=face_rows, replace_existing=True)

    # ── 3. Person matching + centroid update ───────────────────────────────
    persons = fetch_all_person_centroids()   # [(person_id, display_name, centroid_ndarray)]
    face_id_map = fetch_face_ids_by_photo(req.photo_id)  # [(face_id, face_index), ...]
    assigned_person_names: list[str] = []

    for (face_id, face_index), face_row in zip(face_id_map, face_rows):
        embedding = face_row.embedding  # np.float32 (512,)
        pid, pname, dist = match_embedding_to_person(embedding, persons)

        if pid is not None:
            # ── Known person: link face → person, update centroid ──────
            set_face_person_id(face_id, pid)

            all_embs = fetch_embeddings_by_person_id(pid)
            new_centroid = all_embs.mean(axis=0)
            update_person_centroid(pid, new_centroid)

            # Update local cache so subsequent faces in same image benefit
            for i, (p_id, p_name, _) in enumerate(persons):
                if p_id == pid:
                    persons[i] = (p_id, p_name, new_centroid)
                    break

            assigned_person_names.append(pname)
        else:
            # ── Unknown person: create a new Person N entry ────────────
            person_count = get_person_count()
            new_name = f"Person {person_count + 1}"
            new_pid = insert_person(display_name=new_name, centroid_embedding=embedding)

            # Link this face to the new person
            set_face_person_id(face_id, new_pid)

            # Add to local cache so the next face in this image can match it
            persons.append((new_pid, new_name, embedding.copy()))

            assigned_person_names.append(new_name)
            logger.info("Created new person '%s' (id=%d)", new_name, new_pid)

    # ── 4. Post a FACE tag only for persons seen in 2+ distinct photos ─────
    # Background / one-off people stay in the DB for future re-matching
    # but don't pollute the tag list until they recur.
    unique_pid_name = list(dict.fromkeys(zip(assigned_person_ids, assigned_person_names)))
    tags_posted:  list[str] = []
    tags_skipped: list[str] = []

    for pid, name in unique_pid_name:
        photo_count = count_distinct_photos_for_person(pid)
        if photo_count >= 2:
            ok = _post_face_tag_to_spring(req.photo_id, name)
            if ok:
                tags_posted.append(name)
        else:
            tags_skipped.append(name)
            logger.info(
                "Skipping tag for '%s' (person_id=%d) — only in 1 photo so far", name, pid
            )

    return {
        "photo_id": req.photo_id,
        "type": "PEOPLE",
        "faces_detected": len(faces),
        "unique_persons": [n for _, n in unique_pid_name],
        "tags_added": tags_posted,
        "tags_skipped_single_appearance": tags_skipped,
    }


@app.post("/ingest-faces-old")
async def ingest_faces_old(
    file: UploadFile = File(...),
    model: str = Query("insightface"),
):
    detector = face_detectors.get(model)
    if detector is None:
        raise HTTPException(status_code=400, detail="Unknown face detector")

    # Enforce embedding-capable model
    if detector.name != "insightface":
        raise HTTPException(status_code=400, detail="ingest-faces requires an embedding-capable detector (use model=insightface)")

    temp_filename = _save_upload_to_temp(file)
    try:
        image = cv2.imread(temp_filename)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not read image file")

        h, w = image.shape[:2]

        # Stable-ish key: original filename + uuid
        original = file.filename or "upload.jpg"
        s3_key = f"uploads/{uuid.uuid4().hex}_{original}"

        photo_id = upsert_photo_by_sha1(
            sha1=hashlib.sha1(open(temp_filename, "rb").read()).hexdigest(),
            original_filename=original,
            width=w,
            height=h,
        )

        det = detector.detect(image)
        faces = det.get("faces", []) if isinstance(det.get("faces"), list) else []

        # Make ordering stable: sort by (y1, x1)
        faces = sorted(faces, key=lambda f: (f["bbox"][1], f["bbox"][0]))

        face_rows = []
        for i, f in enumerate(faces):
            x1, y1, x2, y2 = f["bbox"]
            face_rows.append(
                FaceRow(
                    face_index=i,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    det_score=float(f["det_score"]),
                    embedding=f["embedding"],
                )
            )

        inserted = insert_faces(photo_id=photo_id, faces=face_rows, replace_existing=True)

        return {
            "photo_id": photo_id,
            "original_filename": original,
            "faces_detected": len(face_rows),
            "db_rows_affected": inserted,
        }
    finally:
        try:
            os.remove(temp_filename)
        except Exception:
            pass