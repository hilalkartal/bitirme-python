import os
import mimetypes
import requests

API_URL = "http://127.0.0.1:8000/ingest-faces"
IMAGE_FOLDER = r"C:\Users\Lenovo\IdeaProjects\bitirme\testimages\people"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

def main():
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

if __name__ == "__main__":
    main()
