# cluster_faces.py
import numpy as np
from sklearn.cluster import DBSCAN

from repository import fetch_face_embeddings_for_clustering, update_face_cluster_ids

def run_dbscan(embs: np.ndarray, eps: float = 0.35, min_samples: int = 3) -> np.ndarray:
    if embs.shape[0] == 0:
        return np.array([], dtype=int)
    return DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(embs)

def cluster_new_faces(limit: int = 50000, eps: float = 0.35, min_samples: int = 3):
    face_ids, embs = fetch_face_embeddings_for_clustering(limit=limit, only_unclustered=True)
    print(f"Loaded {len(face_ids)} faces (unclustered)")

    labels = run_dbscan(embs, eps=eps, min_samples=min_samples)
    updated = update_face_cluster_ids(face_ids, labels.tolist())

    print(f"Updated cluster_id for {updated} faces")

    if len(labels):
        uniq, counts = np.unique(labels, return_counts=True)
        top = sorted(zip(uniq.tolist(), counts.tolist()), key=lambda x: x[1], reverse=True)[:20]
        print("Top clusters (cluster_id -> count):")
        for cid, cnt in top:
            print(f"  {cid}: {cnt}")

if __name__ == "__main__":
    cluster_new_faces(limit=50000, eps=0.35, min_samples=2)