import cv2
import os
import shutil
from pathlib import Path
from person_vs_scenery.yolov8 import YOLOv8Detector

def organize_images(source_folder, output_folder=None):
    """
    FOR TEST PURPOSES:
    
    Organize images from a folder into 'people' and 'scenery' subfolders using YOLOv8.
    
    Args:
        source_folder: Path to the folder containing images
        output_folder: Path to the output folder (if None, creates subfolder in source)
    """
    
    # Initialize detector
    detector = YOLOv8Detector()
    
    # Set output folder
    if output_folder is None:
        output_folder = source_folder
    
    people_folder = os.path.join(output_folder, "people")
    scenery_folder = os.path.join(output_folder, "scenery")
    
    # Create output folders
    os.makedirs(people_folder, exist_ok=True)
    os.makedirs(scenery_folder, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Get all image files
    image_files = [
        f for f in os.listdir(source_folder)
        if os.path.isfile(os.path.join(source_folder, f))
        and Path(f).suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No images found in {source_folder}")
        return
    
    print(f"Found {len(image_files)} images. Processing...")
    print("-" * 50)
    
    people_count = 0
    scenery_count = 0
    
    for idx, filename in enumerate(image_files, 1):
        filepath = os.path.join(source_folder, filename)
        
        try:
            # Read image
            image = cv2.imread(filepath)
            if image is None:
                print(f"[{idx}/{len(image_files)}] {filename}: ERROR - Could not read image")
                continue
            
            # Detect
            result = detector.detect(image)
            
            # Classify
            is_people = result["people"] > 0
            confidence = result["confidence"]
            
            # Determine destination
            if is_people:
                destination = os.path.join(people_folder, filename)
                category = "PEOPLE"
                people_count += 1
            else:
                destination = os.path.join(scenery_folder, filename)
                category = "SCENERY"
                scenery_count += 1
            
            # Copy file
            shutil.copy2(filepath, destination)
            
            print(f"[{idx}/{len(image_files)}] {filename}: {category} (people: {result['people']}, confidence: {confidence})")
            
        except Exception as e:
            print(f"[{idx}/{len(image_files)}] {filename}: ERROR - {str(e)}")
    
    print("-" * 50)
    print(f"✓ Processing complete!")
    print(f"  People images: {people_count}")
    print(f"  Scenery images: {scenery_count}")
    print(f"  Output folder: {output_folder}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python person_or_scenery.py <source_folder> [output_folder]")
        print("\nExample:")
        print("  python person_or_scenery.py ./photos")
        print("  python person_or_scenery.py ./photos ./organized_photos")
        sys.exit(1)
    
    source = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(source):
        print(f"Error: Source folder '{source}' does not exist")
        sys.exit(1)
    
    organize_images(source, output)
