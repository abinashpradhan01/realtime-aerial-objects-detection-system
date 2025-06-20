from ultralytics import YOLO
import os
import cv2
import numpy as np
import torch


class ObjectDetector:
    def __init__(self, model_path="models/2_best.pt", threshold=0.5, device=None):
        """Initialize the object detector with specified model and parameters"""
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            print(f"[INFO] Loading model from: {model_path}")
            self.model = YOLO(model_path)
            
            # Only fuse if model supports it (some models might not)
            try:
                self.model.fuse()
                print("[INFO] Model fused successfully")
            except Exception as e:
                print(f"[WARNING] Could not fuse model: {e}")
            
            self.threshold = threshold
            self.device = device
            self.model_path = model_path
            
            print(f"[SUCCESS] Model loaded successfully on {self.device}")
            print(f"[INFO] Detection threshold: {self.threshold}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def detect_single_image(self, image_path):
        """Run detection on a single image"""
        try:
            if not os.path.exists(image_path):
                print(f"❌ Image file not found: {image_path}")
                return None
            
            # Run inference
            results = self.model(image_path, conf=self.threshold, device=self.device, verbose=False)
            return results
            
        except Exception as e:
            print(f"❌ Error during detection on {image_path}: {e}")
            return None

    def detect_objects_in_folder(self, input_folder="core/input_frames", output_folder="core/output_frames"):
        """Detect objects in all images in the input folder and save annotated images"""
        
        # Create output folder
        try:
            os.makedirs(output_folder, exist_ok=True)
        except Exception as e:
            print(f"❌ Error creating output folder: {e}")
            return

        # Check if input folder exists
        if not os.path.exists(input_folder):
            print(f"❌ Input folder not found: {input_folder}")
            return

        # Get all image files
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
        try:
            image_files = [
                f for f in os.listdir(input_folder) 
                if f.lower().endswith(image_extensions)
            ]
            image_files.sort()  # Sort for consistent processing order
        except Exception as e:
            print(f"❌ Error reading input folder: {e}")
            return
        
        if not image_files:
            print(f"❌ No image files found in: {input_folder}")
            return

        print(f"[INFO] Processing {len(image_files)} images...")
        processed_count = 0
        detection_count = 0

        for filename in image_files:
            image_path = os.path.join(input_folder, filename)
            
            try:
                # Run detection
                results = self.detect_single_image(image_path)
                
                if results is None:
                    print(f"[SKIP] {filename} - Detection failed")
                    continue
                
                # Check if any objects were detected
                if not results or len(results) == 0:
                    print(f"[NO DETECTIONS] {filename} - No results returned")
                    continue
                
                result = results[0]  # Get first result
                
                if result.boxes is None or len(result.boxes) == 0:
                    print(f"[NO DETECTIONS] {filename} - No objects detected")
                    continue

                # Load original image
                original_image = cv2.imread(image_path)
                if original_image is None:
                    print(f"❌ Could not load image: {filename}")
                    continue

                # Get detection boxes
                boxes = result.boxes.data.cpu().numpy()
                
                if len(boxes) == 0:
                    print(f"[NO DETECTIONS] {filename} - Empty boxes")
                    continue

                # Draw bounding boxes and labels
                for i, box in enumerate(boxes):
                    try:
                        x1, y1, x2, y2, conf, cls = box[:6]
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        
                        # Ensure coordinates are within image bounds
                        height, width = original_image.shape[:2]
                        x1 = max(0, min(x1, width - 1))
                        y1 = max(0, min(y1, height - 1))
                        x2 = max(x1 + 1, min(x2, width))
                        y2 = max(y1 + 1, min(y2, height))
                        
                        # Draw bounding box
                        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"Drone: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        
                        # Ensure label fits within image
                        label_y = max(y1 - 10, label_size[1] + 5)
                        
                        cv2.putText(original_image, label, (x1, label_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                    except Exception as e:
                        print(f"⚠️ Warning: Error drawing box {i} for {filename}: {e}")
                        continue

                # Save annotated image
                output_path = os.path.join(output_folder, filename)
                success = cv2.imwrite(output_path, original_image)
                
                if success:
                    detection_count += 1
                    print(f"[SAVED] {filename} -> {output_path} ({len(boxes)} objects)")
                else:
                    print(f"❌ Failed to save: {output_path}")

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                continue
            
            processed_count += 1

        print(f"[SUMMARY] Processed: {processed_count}/{len(image_files)} images")
        print(f"[SUMMARY] Images with detections: {detection_count}")

    def detect_frame_array(self, frame_array):
        """Detect objects in a numpy array frame (for live detection)"""
        try:
            if frame_array is None or frame_array.size == 0:
                return None
            
            # Run inference directly on the numpy array
            results = self.model(frame_array, conf=self.threshold, device=self.device, verbose=False)
            return results
            
        except Exception as e:
            print(f"❌ Error during frame detection: {e}")
            return None

    def annotate_frame(self, frame, results):
        """Annotate frame with detection results"""
        try:
            if results is None or len(results) == 0:
                return frame
            
            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return frame
            
            # Get detection boxes
            boxes = result.boxes.data.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box[:6]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Ensure coordinates are within frame bounds
                height, width = frame.shape[:2]
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(x1 + 1, min(x2, width))
                y2 = max(y1 + 1, min(y2, height))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"Drone: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            print(f"❌ Error annotating frame: {e}")
            return frame