import cv2
import os
import mediapipe as mp
from pathlib import Path


def draw_faces_on_images(source_folder, output_folder=None):
    """
    FOR TEST PURPOSES

    Process images from a folder and draw green boxes around detected faces using MediaPipe.
    
    Args:
        source_folder: Path to the folder containing images
        output_folder: Path to the output folder (if None, creates 'faces_boxed' subfolder in source)

        python draw_face_boxes.py C:\path\to\photos C:\path\to\output
    """
    
    # Initialize MediaPipe Face Detection
    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )
    
    # Set output folder
    if output_folder is None:
        output_folder = os.path.join(source_folder, "faces_boxed")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
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
    
    total_faces = 0
    images_with_faces = 0
    
    for idx, filename in enumerate(image_files, 1):
        filepath = os.path.join(source_folder, filename)
        
        try:
            # Read image
            image = cv2.imread(filepath)
            if image is None:
                print(f"[{idx}/{len(image_files)}] {filename}: ERROR - Could not read image")
                continue
            
            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = detector.process(rgb)
            
            faces_count = 0
            confidences = []
            
            if results.detections:
                h, w, _ = image.shape
                
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convert relative coordinates to pixel coordinates
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    box_w = int(bbox.width * w)
                    box_h = int(bbox.height * h)
                    
                    # Draw green rectangle
                    cv2.rectangle(image, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
                    
                    # Get confidence score
                    confidence = detection.score[0]
                    confidences.append(confidence)
                    
                    # Draw confidence text
                    text = f"{confidence:.2f}"
                    cv2.putText(image, text, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    faces_count += 1
                
                images_with_faces += 1
                total_faces += faces_count
            
            # Save processed image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, image)
            
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            status = f"{faces_count} face(s), avg conf: {avg_conf:.3f}" if faces_count > 0 else "No faces detected"
            print(f"[{idx}/{len(image_files)}] {filename}: {status}")
            
        except Exception as e:
            print(f"[{idx}/{len(image_files)}] {filename}: ERROR - {str(e)}")
    
    detector.close()
    
    print("-" * 50)
    print(f"✓ Processing complete!")
    print(f"  Total images processed: {len(image_files)}")
    print(f"  Images with faces: {images_with_faces}")
    print(f"  Total faces detected: {total_faces}")
    print(f"  Output folder: {output_folder}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python draw_face_boxes.py <source_folder> [output_folder]")
        print("\nExample:")
        print("  python draw_face_boxes.py ./photos")
        print("  python draw_face_boxes.py ./photos ./faces_detected")
        sys.exit(1)
    
    source = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(source):
        print(f"Error: Source folder '{source}' does not exist")
        sys.exit(1)
    
    draw_faces_on_images(source, output)
