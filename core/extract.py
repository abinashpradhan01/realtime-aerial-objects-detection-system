import cv2
import os
import shutil
import tempfile


def get_free_disk_space(path):
    """Get free disk space in bytes for the given path"""
    try:
        total, used, free = shutil.disk_usage(path)
        return free
    except Exception as e:
        print(f"Warning: Could not check disk space: {e}")
        return float('inf')  # Assume unlimited space if check fails


def estimate_frame_size(width, height):
    """Estimate size of a single frame in bytes (JPEG compression)"""
    return int(width * height * 0.1)  # Approx. 0.1 bytes per pixel for JPEG


def extractFrames(video_path: str, max_frames=180):
    """Extract frames from video with disk space checking and frame limits"""
    print(f"[INFO] Extracting frames from: {video_path}")
    
    output_folder = "core/input_frames"
    
    # Create output folder if it doesn't exist
    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception as e:
        print(f"❌ Error creating output folder: {e}")
        return 0
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        return 0
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_path}")
        return 0
    
    try:
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps <= 0:
            print("⚠️ Warning: FPS not detected. Defaulting to 30 FPS.")
            fps = 30
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"[INFO] Video properties - Width: {width}, Height: {height}")
        print(f"[INFO] FPS: {fps:.2f}, Total Frames: {frame_count}, Duration: {duration:.2f}s")
        
        # Check free space
        frame_size = estimate_frame_size(width, height)
        required_space = frame_size * min(max_frames, int(duration))  # Don't exceed video duration
        free_space = get_free_disk_space(output_folder)
        
        if free_space < required_space:
            print(f"❌ Error: Insufficient disk space.")
            print(f"Required: {required_space/1024/1024:.1f}MB, Available: {free_space/1024/1024:.1f}MB")
            cap.release()
            return 0
        
        # Calculate frame interval (1 frame per second)
        frame_interval = max(1, int(fps))  # Ensure at least 1 frame interval
        current_frame = 0
        saved_frame_count = 0
        
        print(f"[INFO] Extracting 1 frame per second (interval: {frame_interval} frames)")
        
        while current_frame < frame_count and saved_frame_count < max_frames:
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            
            if not ret:
                print(f"⚠️ Warning: Could not read frame at position {current_frame}")
                break
            
            # Validate frame
            if frame is None or frame.size == 0:
                print(f"⚠️ Warning: Empty frame at position {current_frame}")
                current_frame += frame_interval
                continue
            
            # Save frame
            frame_filename = f"frame_{saved_frame_count:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            
            try:
                success = cv2.imwrite(frame_path, frame)
                if success:
                    saved_frame_count += 1
                    if saved_frame_count % 10 == 0:
                        print(f"[INFO] Saved {saved_frame_count} frames...")
                else:
                    print(f"⚠️ Warning: Failed to save frame {saved_frame_count}")
            except Exception as e:
                print(f"❌ Error saving frame {saved_frame_count}: {e}")
            
            current_frame += frame_interval
        
        cap.release()
        print(f"[SUCCESS] Extracted {saved_frame_count} frames to '{output_folder}'")
        return saved_frame_count
        
    except Exception as e:
        print(f"❌ Error during frame extraction: {e}")
        cap.release()
        return 0