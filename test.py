import os
import shutil
import mimetypes
import requests

from db import get_conn

API_URL = "http://127.0.0.1:8000/ingest-faces"
IMAGE_FOLDER = r"C:\Users\Lenovo\IdeaProjects\bitirme\testimages\people"
OUTPUT_FOLDER = r"C:\Users\Lenovo\IdeaProjects\bitirme\testimages\people_clusters"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

def t_save_images_to_db():
    if not os.path.exists(IMAGE_FOLDER):
        raise FileNotFoundError(f"Folder not found: {IMAGE_FOLDER}")

    image_files = [
        f for f in os.listdir(IMAGE_FOLDER)
        if os.path.isfile(os.path.join(IMAGE_FOLDER, f))
        and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        print(f"No image files found in {IMAGE_FOLDER}")
        return

    print(f"Found {len(image_files)} image(s)\n")

    for idx, filename in enumerate(image_files, 1):
        image_path = os.path.join(IMAGE_FOLDER, filename)
        mime = mimetypes.guess_type(image_path)[0] or "application/octet-stream"

        print(f"[{idx}/{len(image_files)}] {filename} ({mime})")

        with open(image_path, "rb") as f:
            files = {"file": (filename, f, mime)}
            params = {"model": "insightface"}

            try:
                # (connect timeout, read timeout)
                resp = requests.post(API_URL, files=files, params=params, timeout=(5, 300))
                print("  Status:", resp.status_code)
                print("  Body:", resp.text[:500])
            except requests.exceptions.ReadTimeout:
                print("  ❌ ReadTimeout: server is not responding in time")
            except requests.exceptions.ConnectTimeout:
                print("  ❌ ConnectTimeout: cannot reach server")
            except requests.exceptions.ConnectionError as e:
                print("  ❌ ConnectionError:", e)
            except Exception as e:
                print("  ❌ Error:", e)

        print()


def t_create_people_folders():
    """Query faces with assigned persons and copy images into per-person folders."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT p.id, p.display_name, i.image_path
            FROM faces f
            JOIN images i ON i.id = f.image_id
            JOIN persons p ON p.id = f.person_id
            WHERE f.person_id IS NOT NULL
            ORDER BY p.id
        """)
        rows = cur.fetchall()
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()

    if not rows:
        print("No faces with assigned persons found.")
        return

    # Group image filenames by (person_id, display_name)
    persons = {}
    for person_id, display_name, image_path in rows:
        persons.setdefault((person_id, display_name), set()).add(image_path)

    print(f"Found {len(persons)} person(s)\n")

    # Create output folder (clean start)
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)

    for (person_id, display_name), filenames in sorted(persons.items()):
        safe_name = display_name.replace(" ", "_")
        person_dir = os.path.join(OUTPUT_FOLDER, safe_name)
        os.makedirs(person_dir, exist_ok=True)

        for filename in sorted(filenames):
            src = os.path.join(IMAGE_FOLDER, filename)
            if not os.path.isfile(src):
                print(f"  WARNING: {filename} not found in {IMAGE_FOLDER}")
                continue
            shutil.copy2(src, person_dir)

        print(f"  {safe_name}: {len(filenames)} image(s)")

    print(f"\nDone. Output: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    # t_save_images_to_db()          # ingest images via API
    t_create_people_folders()  # copy images into per-cluster folders
