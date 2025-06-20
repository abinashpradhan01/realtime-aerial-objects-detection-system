import streamlit as st
import os
import shutil
import tempfile
import cv2
import numpy as np
from core.extract import extractFrames
from core.predict import ObjectDetector
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import time

# MUST BE FIRST - Set page config
st.set_page_config(
    page_title="Drone Detection App",
    page_icon="🚁",
    layout="wide"
)

# Model paths
VIDEO_MODEL_PATH = "models/2_best.pt"          # For stock footage
LIVE_MODEL_PATH = "models/nano/1_nano.pt"      # For live webcam

# Directory paths
INPUT_FRAMES_DIR = "core/input_frames"
OUTPUT_FRAMES_DIR = "core/output_frames"

# Initialize detectors with error handling
@st.cache_resource
def load_video_detector():
    """Load video detector with caching"""
    try:
        if not os.path.exists(VIDEO_MODEL_PATH):
            return None
        return ObjectDetector(model_path=VIDEO_MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading video model: {e}")
        return None

@st.cache_resource
def load_live_detector():
    """Load live detector with caching"""
    try:
        if not os.path.exists(LIVE_MODEL_PATH):
            return None
        return ObjectDetector(model_path=LIVE_MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading live model: {e}")
        return None

# Global detector instances
video_detector = load_video_detector()
live_detector = load_live_detector()

class DroneDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = live_detector
        self.threshold = self.model.threshold if self.model else 0.5
        self.frame_count = 0
        self.detection_count = 0

    def recv(self, frame):
        try:
            if self.model is None:
                return frame
            
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            
            # Run detection
            results = self.model.detect_frame_array(img)
            
            # Annotate frame if objects detected
            if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                img = self.model.annotate_frame(img, results)
                self.detection_count += 1
            
            self.frame_count += 1
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            st.error(f"Error in video processing: {e}")
            return frame

def clear_folder(folder):
    """Clear and recreate folder"""
    try:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
    except Exception as e:
        st.error(f"Error clearing folder {folder}: {e}")

def cleanup_temp_files(paths):
    """Clean up temporary files"""
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            st.warning(f"Could not delete temporary file {path}: {e}")

def process_uploaded_video(video_file):
    """Process uploaded video file for drone detection"""
    if video_detector is None:
        return [], 'Video detector not loaded. Please check model file.'
    
    # Create temporary video file
    temp_video_path = tempfile.mktemp(suffix='.mp4')
    
    try:
        # Save uploaded file to temporary location
        with open(temp_video_path, 'wb') as f:
            f.write(video_file.read())
        
        # Clear input and output directories
        clear_folder(INPUT_FRAMES_DIR)
        clear_folder(OUTPUT_FRAMES_DIR)
        
        # Extract frames
        st.info("Extracting frames from video...")
        frame_count = extractFrames(temp_video_path)
        
        if frame_count == 0:
            cleanup_temp_files([temp_video_path])
            return [], 'No frames could be extracted from the video.'
        
        st.info(f"Extracted {frame_count} frames. Running drone detection...")
        
        # Run detection on extracted frames
        video_detector.detect_objects_in_folder(INPUT_FRAMES_DIR, OUTPUT_FRAMES_DIR)
        
        # Get detected images
        try:
            detected_files = [
                f for f in os.listdir(OUTPUT_FRAMES_DIR)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]
            detected_files.sort()  # Sort for consistent order
            
            detected_images = []
            for filename in detected_files:
                img_path = os.path.join(OUTPUT_FRAMES_DIR, filename)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        detected_images.append(img_rgb)
                except Exception as e:
                    st.warning(f"Could not load image {filename}: {e}")
            
        except Exception as e:
            st.error(f"Error reading output directory: {e}")
            detected_images = []
        
        # Cleanup
        cleanup_temp_files([temp_video_path])
        
        if not detected_images:
            return [], f'No drones detected in {frame_count} frames processed.'
        
        return detected_images, f'Detection complete: {len(detected_images)} frame(s) with drone detections out of {frame_count} total frames.'
        
    except Exception as e:
        cleanup_temp_files([temp_video_path])
        st.error(f"Error processing video: {e}")
        return [], f'Error processing video: {str(e)}'

def main():
    st.title('🚁 Drone Detection App')
    st.markdown('Upload a video or use live webcam to detect drones using custom YOLOv11 models.')
    
    # Sidebar for mode selection
    st.sidebar.header("Detection Mode")
    mode = st.sidebar.selectbox(
        'Choose Detection Mode', 
        ['Video Upload Detection', 'Live Webcam Detection']
    )
    
    # Model status
    st.sidebar.header("Model Status")
    video_status = "✅ Loaded" if video_detector else "❌ Not Loaded"
    live_status = "✅ Loaded" if live_detector else "❌ Not Loaded"
    st.sidebar.write(f"Video Model: {video_status}")
    st.sidebar.write(f"Live Model: {live_status}")
    
    if mode == 'Video Upload Detection':
        st.header('📹 Video Upload Detection')
        st.write('Upload any video file (stock footage, webcam recordings, etc.). The app will extract frames and detect drones.')
        
        if video_detector is None:
            st.error("Video detection model not available. Please check that the model file exists at: " + VIDEO_MODEL_PATH)
            return
        
        video_file = st.file_uploader(
            'Upload Video File', 
            type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
            help="Supported formats: MP4, AVI, MOV, MKV, WMV. Works with any video source including webcam recordings."
        )
        
        if video_file is not None:
            # Display video info
            st.info(f"📁 File: {video_file.name} ({video_file.size / (1024*1024):.1f} MB)")
            
            if st.button("🔍 Start Detection", type="primary"):
                with st.spinner('Processing video... This may take a few minutes.'):
                    images, status = process_uploaded_video(video_file)
                
                st.success(status)
                
                if images:
                    st.subheader(f"🎯 Detection Results ({len(images)} frames)")
                    
                    # Display images in columns
                    cols_per_row = 3
                    for i in range(0, len(images), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            if i + j < len(images):
                                with col:
                                    st.image(
                                        images[i + j], 
                                        caption=f'Frame {i + j + 1}',
                                        use_container_width=True
                                    )
    
    elif mode == 'Live Webcam Detection':
        st.header('📷 Live Webcam Detection')
        st.write('Click "START" to begin real-time drone detection using your webcam.')
        
        if live_detector is None:
            st.error("Live detection model not available. Please check that the model file exists at: " + LIVE_MODEL_PATH)
            return
        
        # Instructions
        st.info("💡 **Instructions:**\n"
                "1. Click 'START' to activate your webcam\n"
                "2. Allow camera permissions when prompted\n"
                "3. Drones will be highlighted with green bounding boxes\n"
                "4. Click 'STOP' to end the session")
        
        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="live_detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=DroneDetectionProcessor,
            media_stream_constraints={
                "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
                "audio": False
            },
            async_processing=True,
        )
        
        # Display stats if active
        if webrtc_ctx.video_processor:
            st.subheader("📊 Detection Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Frames Processed", webrtc_ctx.video_processor.frame_count)
            with col2:
                st.metric("Detections", webrtc_ctx.video_processor.detection_count)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Drone Detection App**")
    st.sidebar.markdown("Powered by YOLOv11 & Streamlit")

if __name__ == "__main__":
    main()