# cluster_faces.py
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

from repository import (
    fetch_face_embeddings_for_clustering,
    update_face_cluster_ids,
    fetch_all_person_centroids,
    insert_person,
    set_faces_person_id,
    fetch_embeddings_by_person_id,
    update_person_centroid,
    get_person_count,
)


def run_dbscan(embs: np.ndarray, eps: float = 0.35, min_samples: int = 3) -> np.ndarray:
    if embs.shape[0] == 0:
        return np.array([], dtype=int)
    return DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(embs)


def assign_clusters_to_persons(face_ids, labels, embs, eps=0.35):
    """
    For each DBSCAN cluster (label != -1):
      1. Compute cluster centroid
      2. Compare against existing person centroids
      3. If cosine distance < eps: assign to that person
      4. Else: create a new person
      5. Recompute person centroid from all assigned faces
    """
    unique_labels = set(labels.tolist())
    unique_labels.discard(-1)

    if not unique_labels:
        print("No clusters to assign to persons.")
        return

    existing_persons = fetch_all_person_centroids()
    person_count = get_person_count()

    for cluster_label in sorted(unique_labels):
        mask = labels == cluster_label
        cluster_face_ids = [fid for fid, m in zip(face_ids, mask) if m]
        cluster_embs = embs[mask]

        centroid = cluster_embs.mean(axis=0)

        matched_person_id = None
        if existing_persons:
            person_centroids = np.stack([p[2] for p in existing_persons])
            dists = cosine_distances(centroid.reshape(1, -1), person_centroids)[0]
            best_idx = int(np.argmin(dists))
            best_dist = float(dists[best_idx])

            if best_dist < eps:
                matched_person_id = existing_persons[best_idx][0]
                print(f"  Cluster {cluster_label} ({len(cluster_face_ids)} faces) -> "
                      f"existing {existing_persons[best_idx][1]} (dist={best_dist:.3f})")

        if matched_person_id is None:
            person_count += 1
            display_name = f"Person {person_count}"
            person_id = insert_person(display_name, centroid)
            print(f"  Cluster {cluster_label} ({len(cluster_face_ids)} faces) -> "
                  f"new {display_name} (id={person_id})")
            existing_persons.append((person_id, display_name, centroid))
        else:
            person_id = matched_person_id

        set_faces_person_id(cluster_face_ids, person_id)

        # Recompute centroid from ALL faces now belonging to this person
        all_embs = fetch_embeddings_by_person_id(person_id)
        new_centroid = all_embs.mean(axis=0)
        update_person_centroid(person_id, new_centroid)

        # Update local cache
        for i, (pid, name, _) in enumerate(existing_persons):
            if pid == person_id:
                existing_persons[i] = (pid, name, new_centroid)
                break


def cluster_new_faces(limit: int = 50000, eps: float = 0.35, min_samples: int = 3):
    face_ids, embs = fetch_face_embeddings_for_clustering(limit=limit, only_unclustered=True)
    print(f"Loaded {len(face_ids)} faces (unclustered)")

    if len(face_ids) == 0:
        print("Nothing to cluster.")
        return

    labels = run_dbscan(embs, eps=eps, min_samples=min_samples)
    updated = update_face_cluster_ids(face_ids, labels.tolist())

    print(f"Updated cluster_id for {updated} faces")

    if len(labels):
        uniq, counts = np.unique(labels, return_counts=True)
        top = sorted(zip(uniq.tolist(), counts.tolist()), key=lambda x: x[1], reverse=True)[:20]
        print("Top clusters (cluster_id -> count):")
        for cid, cnt in top:
            print(f"  {cid}: {cnt}")

    print("\nAssigning clusters to persons...")
    assign_clusters_to_persons(face_ids, labels, embs, eps=eps)
    print("Person assignment complete.")


if __name__ == "__main__":
    cluster_new_faces(limit=50000, eps=0.35, min_samples=2)
