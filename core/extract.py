import cv2
import os
import shutil
import tempfile
import math


def get_free_disk_space(path):
    """Get free disk space in bytes for the given path"""
    try:
        total, used, free = shutil.disk_usage(path)
        return free
    except Exception as e:
        print(f"Warning: Could not check disk space: {e}")
        return float('inf')  # Assume unlimited space if check fails


def estimate_frame_size(width, height, quality_factor=0.1):
    """Estimate size of a single frame in bytes (JPEG compression)"""
    # More accurate estimation based on resolution
    base_size = width * height * 3  # RGB channels
    compressed_size = base_size * quality_factor
    return int(compressed_size)


def validate_video_file(video_path):
    """Validate video file and return basic properties"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Cannot open video file"
        
        # Get basic properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.release()
        
        # Validate properties
        if width <= 0 or height <= 0:
            return None, "Invalid video dimensions"
        
        if fps <= 0:
            fps = 30  # Default fallback
            print("⚠️ Warning: FPS not detected. Using default 30 FPS.")
        
        if frame_count <= 0:
            return None, "Invalid frame count"
        
        duration = frame_count / fps
        
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration
        }, None
        
    except Exception as e:
        return None, f"Error validating video: {str(e)}"


def calculate_extraction_params(video_props, max_frames=180, target_fps=1):
    """Calculate optimal frame extraction parameters"""
    fps = video_props["fps"]
    duration = video_props["duration"]
    frame_count = video_props["frame_count"]
    
    # Calculate frame interval for target FPS (1 frame per second by default)
    frame_interval = max(1, int(fps / target_fps))
    
    # Calculate expected number of frames
    expected_frames = min(max_frames, int(duration * target_fps))
    
    # Adjust if video is very short
    if expected_frames < 1:
        expected_frames = min(max_frames, frame_count)
        frame_interval = max(1, frame_count // expected_frames)
    
    return {
        "frame_interval": frame_interval,
        "expected_frames": expected_frames,
        "target_fps": target_fps
    }


def extractFrames(video_path: str, max_frames=180, target_fps=1):
    """Extract frames from video with improved error handling and optimization"""
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
    
    # Validate video file
    video_props, error = validate_video_file(video_path)
    if video_props is None:
        print(f"❌ Error: {error}")
        return 0
    
    print(f"[INFO] Video properties - Width: {video_props['width']}, Height: {video_props['height']}")
    print(f"[INFO] FPS: {video_props['fps']:.2f}, Total Frames: {video_props['frame_count']}, Duration: {video_props['duration']:.2f}s")
    
    # Calculate extraction parameters
    extraction_params = calculate_extraction_params(video_props, max_frames, target_fps)
    
    print(f"[INFO] Extraction plan - Interval: {extraction_params['frame_interval']}, Expected frames: {extraction_params['expected_frames']}")
    
    # Check disk space
    frame_size = estimate_frame_size(video_props['width'], video_props['height'])
    required_space = frame_size * extraction_params['expected_frames']
    free_space = get_free_disk_space(output_folder)
    
    if free_space < required_space:
        print(f"❌ Error: Insufficient disk space.")
        print(f"Required: {required_space/1024/1024:.1f}MB, Available: {free_space/1024/1024:.1f}MB")
        return 0
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_path}")
        return 0
    
    try:
        frame_interval = extraction_params['frame_interval']
        current_frame = 0
        saved_frame_count = 0
        max_expected = extraction_params['expected_frames']
        
        print(f"[INFO] Extracting ~{target_fps} frame(s) per second (interval: {frame_interval} frames)")
        
        while current_frame < video_props['frame_count'] and saved_frame_count < max_frames:
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            
            if not ret:
                print(f"⚠️ Warning: Could not read frame at position {current_frame}")
                # Try to continue with next frame
                current_frame += frame_interval
                continue
            
            # Validate frame
            if frame is None or frame.size == 0:
                print(f"⚠️ Warning: Empty frame at position {current_frame}")
                current_frame += frame_interval
                continue
            
            # Validate frame dimensions
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                print(f"⚠️ Warning: Invalid frame format at position {current_frame}")
                current_frame += frame_interval
                continue
            
            # Generate frame filename with zero-padding
            frame_filename = f"frame_{saved_frame_count:06d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            
            try:
                # Use optimized JPEG settings for better compression and quality
                jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 85, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
                success = cv2.imwrite(frame_path, frame, jpeg_params)
                
                if success and os.path.exists(frame_path):
                    saved_frame_count += 1
                    if saved_frame_count % 10 == 0 or saved_frame_count <= 5:
                        print(f"[INFO] Saved {saved_frame_count} frames...")
                else:
                    print(f"⚠️ Warning: Failed to save frame {saved_frame_count}")
                    
            except Exception as e:
                print(f"❌ Error saving frame {saved_frame_count}: {e}")
            
            current_frame += frame_interval
        
        cap.release()
        
        # Final validation
        if saved_frame_count == 0:
            print(f"❌ Error: No frames were successfully extracted")
            return 0
        
        # Verify saved files
        try:
            saved_files = [f for f in os.listdir(output_folder) if f.lower().endswith('.jpg')]
            actual_count = len(saved_files)
            
            if actual_count != saved_frame_count:
                print(f"⚠️ Warning: Expected {saved_frame_count} files, found {actual_count}")
                saved_frame_count = actual_count
                
        except Exception as e:
            print(f"⚠️ Warning: Could not verify saved files: {e}")
        
        print(f"[SUCCESS] Extracted {saved_frame_count} frames to '{output_folder}'")
        
        # Performance summary
        extraction_rate = saved_frame_count / video_props['duration'] if video_props['duration'] > 0 else 0
        print(f"[INFO] Extraction rate: {extraction_rate:.2f} frames/second of video")
        
        return saved_frame_count
        
    except Exception as e:
        print(f"❌ Error during frame extraction: {e}")
        cap.release()
        return 0
    
    finally:
        # Ensure video capture is always released
        if cap.isOpened():
            cap.release()


def cleanup_extracted_frames(folder_path="core/input_frames"):
    """Clean up extracted frames folder"""
    try:
        if os.path.exists(folder_path):
            file_count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            shutil.rmtree(folder_path)
            print(f"[INFO] Cleaned up {file_count} extracted frames")
            return True
    except Exception as e:
        print(f"⚠️ Warning: Could not cleanup frames folder: {e}")
        return False


def get_extraction_info(video_path, max_frames=180, target_fps=1):
    """Get information about what extraction would produce without actually extracting"""
    if not os.path.exists(video_path):
        return None, "Video file not found"
    
    video_props, error = validate_video_file(video_path)
    if video_props is None:
        return None, error
    
    extraction_params = calculate_extraction_params(video_props, max_frames, target_fps)
    
    info = {
        "video_duration": video_props['duration'],
        "video_fps": video_props['fps'],
        "total_frames": video_props['frame_count'],
        "extraction_interval": extraction_params['frame_interval'],
        "expected_extracted_frames": extraction_params['expected_frames'],
        "estimated_size_mb": (estimate_frame_size(video_props['width'], video_props['height']) * 
                             extraction_params['expected_frames']) / (1024 * 1024)
    }
    
    return info, None