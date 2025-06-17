import streamlit as st
from ultralytics import YOLO
import os
import torch
import base64
from utils.image_utils import detect_and_annotate_image
from utils.live_utils import run_live_detection
from utils.video_utils import detect_and_annotate_video

# Disable Streamlit's file watcher to prevent unnecessary reloads
os.environ["STREAMLIT_WATCH_DISABLE"] = "true"

class DummyPath:
    """Dummy class to inject a path attribute for torch.classes to prevent errors."""
    _path = []

# Inject dummy __path__ for torch.classes to prevent potential errors
torch.classes.__path__ = DummyPath()

# === CONFIG ===
# Set the page configuration for the Streamlit app
st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")

# === Load your trained YOLOv8 model ===
# Initialize the YOLO model with the trained weights
model = YOLO("model/best.pt")

# === UI Tabs ===
# Create tabs for different functionalities
tabs = st.tabs(["📷 Annotate Image", "🎥 Annotate Video", "📡 Live Camera"])

# === Image Annotation Tab ===
with tabs[0]:
    st.header("📷 Upload an Image for Annotation")
    # File uploader for image files
    uploaded_img = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        # Process the uploaded image and get the results
        result_img, detected_data, original_img = detect_and_annotate_image(model, uploaded_img)

        st.subheader("📸 Side-by-Side Comparison")
        col1, col2 = st.columns(2)
        with col1:
            # Display the original image
            st.image(original_img, caption="Original Image", use_container_width=True)
        with col2:
            # Display the annotated image
            st.image(result_img, caption="Annotated Image", use_container_width=True)

        if detected_data:
            # Display the detected objects in a table
            st.subheader("📋 Detected Objects")
            st.table(detected_data)
        else:
            st.info("No objects detected.")

# === Video Annotation Tab ===
with tabs[1]:
    st.header("🎥 Upload a Video for Annotation")
    # File uploader for video files
    uploaded_vid = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])
    if uploaded_vid:
        st.info("Processing video. This may take a few seconds...")
        # Process the uploaded video and get the output path
        output_path = detect_and_annotate_video(model, uploaded_vid)

        st.success("Video processed! Here's the result:")
        with open(output_path, "rb") as f:
            video_bytes = f.read()
            # Encode video to base64 for embedding in HTML
            b64_encoded = base64.b64encode(video_bytes).decode()

            # Display the processed video in an HTML video player
            st.markdown(
                f"""
                <video controls width="480" height="370" style="border: 1px solid #ccc; border-radius: 10px; display: block; margin: auto;">
                    <source src="data:video/mp4;base64,{b64_encoded}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                """,
                unsafe_allow_html=True
            )

            st.caption(f"Video loaded successfully! Size: {len(video_bytes) / 1_000_000:.2f} MB")

# === Live Annotation Tab ===
with tabs[2]:
    st.header("📡 Live Annotation from Camera")

    # Initialize session state for live camera if not already
    if "live" not in st.session_state:
        st.session_state.live = False

    # Start and Stop buttons for live camera
    col1, col2 = st.columns(2)
    if col1.button("▶️ Start Live Camera"):
        st.session_state.live = True
        # Run live detection using the model
        run_live_detection(model)

    if col2.button("⏹️ Stop Live Camera"):
        st.session_state.live = False
