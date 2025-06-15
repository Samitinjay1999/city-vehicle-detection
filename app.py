import streamlit as st
from ultralytics import YOLO
import os
import torch
import base64
from utils.image_utils import detect_and_annotate_image
from utils.live_utils import run_live_detection
from utils.video_utils import detect_and_annotate_video
os.environ["STREAMLIT_WATCH_DISABLE"] = "true"

class DummyPath:
    _path = []

# Inject dummy __path__ for torch.classes to prevent error
torch.classes.__path__ = DummyPath()
# === CONFIG ===
st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")

# === Load your trained YOLOv8 model ===
model = YOLO("model/best.pt")  

# === UI Tabs ===
tabs = st.tabs(["📷 Annotate Image", "🎥 Annotate Video", "📡 Live Camera"])

# === Image Annotation ===
with tabs[0]:
    st.header("📷 Upload an Image for Annotation")
    uploaded_img = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        result_img, detected_data = detect_and_annotate_image(model,uploaded_img)
        st.image(result_img, caption="Annotated Image", use_container_width=True)

        if detected_data:
            st.subheader("📋 Detected Objects")
            st.table(detected_data)
        else:
            st.info("No objects detected.")

# === Video Annotation ===
with tabs[1]:
    st.header("🎥 Upload a Video for Annotation")
    uploaded_vid = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])
    if uploaded_vid:
        st.info("Processing video. This may take a few seconds...")
        output_path = detect_and_annotate_video(model,uploaded_vid)

        st.success("Video processed! Here's the result:")
        with open(output_path, "rb") as f:
            video_bytes = f.read()
            b64_encoded = base64.b64encode(video_bytes).decode()  # Encode video to base64

            st.markdown(  # Display HTML video player
                f"""
                <video controls width="480" height="370" style="border: 1px solid #ccc; border-radius: 10px; display: block; margin: auto;">
                    <source src="data:video/mp4;base64,{b64_encoded}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                """,
                unsafe_allow_html=True
            )

            st.caption(f"Video loaded successfully! Size: {len(video_bytes) / 1_000_000:.2f} MB")

# === Live Annotation ===
with tabs[2]:
    st.header("📡 Live Annotation from Camera")

    # Init session state if not already
    if "live" not in st.session_state:
        st.session_state.live = False

    # Start / Stop buttons
    col1, col2 = st.columns(2)
    if col1.button("▶️ Start Live Camera"):
        st.session_state.live = True
        run_live_detection(model)

    if col2.button("⏹️ Stop Live Camera"):
        st.session_state.live = False

