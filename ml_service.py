from fastapi import FastAPI, UploadFile, File, Query, HTTPException
import cv2
import uuid
import os
import shutil
import hashlib
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from repository import insert_faces, FaceRow
from repository import upsert_photo_by_sha1
from repository import (
    fetch_all_person_centroids,
    fetch_face_ids_by_photo,
    set_face_person_id,
    fetch_embeddings_by_person_id,
    update_person_centroid,
)

from person_vs_scenery.haar import HaarDetector
from person_vs_scenery.hog import HOGDetector
from person_vs_scenery.yolov8 import YOLOv8Detector

from face_detectors.mediapipe_face import MediaPipeFaceDetector
from face_detectors.insightface_face import InsightFaceDetector

app = FastAPI()

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