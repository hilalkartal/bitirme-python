# repository.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List
import hashlib
import numpy as np

from db import get_conn


@dataclass
class FaceRow:
    face_index: int
    bbox: Tuple[int, int, int, int]          # (x1, y1, x2, y2)
    embedding: np.ndarray                    # shape (D,), dtype float32 recommended
    det_score: float


def _sha1_of_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def upsert_image(image_path: str, width: Optional[int], height: Optional[int], sha1: Optional[str] = None) -> int:
    """
    Insert or update images row by unique image_path (and optionally sha1).
    Returns the image_id.
    """
    conn = get_conn()
    try:
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO images (image_path, sha1, width, height)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
              sha1  = COALESCE(VALUES(sha1), sha1),
              width = VALUES(width),
              height= VALUES(height)
            """,
            (image_path, sha1, width, height),
        )

        cur.execute("SELECT id FROM images WHERE image_path = %s", (image_path,))
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Upsert failed: could not fetch images.id back")
        (image_id,) = row

        conn.commit()
        return int(image_id)
    except Exception:
        conn.rollback()
        raise
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()


def insert_faces(image_id: int, faces: Sequence[FaceRow], replace_existing: bool = True) -> int:
    """
    Inserts face rows for a given image_id.
    Uses UNIQUE (image_id, face_index).
    """
    if not faces:
        return 0

    conn = get_conn()
    try:
        cur = conn.cursor()

        values: List[tuple] = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox

            emb = np.asarray(f.embedding)
            if emb.ndim != 1:
                raise ValueError(f"Embedding must be 1D, got shape {emb.shape}")
            if emb.dtype != np.float32:
                emb = emb.astype(np.float32)

            emb_bytes = emb.tobytes(order="C")

            values.append((
                image_id,
                int(f.face_index),
                int(x1), int(y1), int(x2), int(y2),
                float(f.det_score),
                emb_bytes,
            ))

        if replace_existing:
            sql = """
                INSERT INTO faces
                  (image_id, face_index, x1, y1, x2, y2, det_score, embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                  x1=VALUES(x1), y1=VALUES(y1), x2=VALUES(x2), y2=VALUES(y2),
                  det_score=VALUES(det_score),
                  embedding=VALUES(embedding)
            """
        else:
            sql = """
                INSERT IGNORE INTO faces
                  (image_id, face_index, x1, y1, x2, y2, det_score, embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """

        cur.executemany(sql, values)
        conn.commit()
        return cur.rowcount
    except Exception:
        conn.rollback()
        raise
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()


def get_face_embedding(face_id: int) -> np.ndarray:

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT embedding FROM faces WHERE id=%s", (face_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"No face with id={face_id}")
        (blob,) = row
        emb = np.frombuffer(blob, dtype=np.float32)  # 512 floats
        return emb
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()


def upsert_image_by_sha1(
    sha1: str,
    image_path: str,
    width: Optional[int],
    height: Optional[int],
) -> int:
    """
    Idempotent upsert:
    - same sha1 => same image row
    """
    conn = get_conn()
    try:
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO images (image_path, sha1, width, height)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
              image_path = VALUES(image_path),
              width = VALUES(width),
              height = VALUES(height)
            """,
            (image_path, sha1, width, height),
        )

        cur.execute("SELECT id FROM images WHERE sha1 = %s", (sha1,))
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Upsert failed: could not fetch images.id by sha1")
        (image_id,) = row

        conn.commit()
        return int(image_id)
    except Exception:
        conn.rollback()
        raise
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()



def fetch_face_embeddings_for_clustering(limit: int = 50000, only_unclustered: bool = True) -> Tuple[List[int], np.ndarray]:
    """
    Returns:
      face_ids: list[int]
      embs: (N, 512) float32
    """
    conn = get_conn()
    try:
        cur = conn.cursor()
        if only_unclustered:
            cur.execute(
                """
                SELECT id, embedding
                FROM faces
                WHERE cluster_id IS NULL
                LIMIT %s
                """,
                (limit,),
            )
        else:
            cur.execute(
                """
                SELECT id, embedding
                FROM faces
                LIMIT %s
                """,
                (limit,),
            )

        rows = cur.fetchall()
        face_ids: List[int] = []
        embs_list: List[np.ndarray] = []

        for face_id, blob in rows:
            face_ids.append(int(face_id))
            emb = np.frombuffer(blob, dtype=np.float32)
            if emb.shape[0] != 512:
                raise ValueError(f"face_id={face_id} has embedding dim {emb.shape[0]} (expected 512)")
            embs_list.append(emb)

        if not embs_list:
            return face_ids, np.zeros((0, 512), dtype=np.float32)

        embs = np.stack(embs_list).astype(np.float32)
        return face_ids, embs
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()


def update_face_cluster_ids(face_ids: List[int], cluster_ids: List[int]) -> int:
    """
    Bulk update faces.cluster_id by face id.
    Returns affected rowcount.
    """
    if len(face_ids) != len(cluster_ids):
        raise ValueError("face_ids and cluster_ids length mismatch")

    if not face_ids:
        return 0

    conn = get_conn()
    try:
        cur = conn.cursor()
        data = [(int(cid), int(fid)) for fid, cid in zip(face_ids, cluster_ids)]
        cur.executemany("UPDATE faces SET cluster_id=%s WHERE id=%s", data)
        conn.commit()
        return cur.rowcount
    except Exception:
        conn.rollback()
        raise
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()