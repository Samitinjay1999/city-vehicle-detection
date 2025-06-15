import streamlit as st
import cv2
import tempfile
import os
from PIL import Image
from ultralytics import YOLO
import numpy as np
import sys
import types
import os
import base64
os.environ["STREAMLIT_WATCH_DISABLE"] = "true"
# Workaround to avoid Streamlit inspecting torch.classes (causes RuntimeError)
import torch

class DummyPath:
    _path = []

# Inject dummy __path__ for torch.classes to prevent error
torch.classes.__path__ = DummyPath()
# === CONFIG ===
st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")

# === Load your trained YOLOv8 model ===
model = YOLO("model/best.pt")  

# === Helper function ===
def detect_and_annotate_image(uploaded_image):
    image = Image.open(uploaded_image).convert("RGB")
    results = model.predict(source=np.array(image), conf=0.25)
    annotated_img = results[0].plot()

    detections = results[0].boxes
    classes = results[0].names

    # Prepare a list of detected objects and their confidence
    detected_data = []
    for box in detections:
        cls_id = int(box.cls[0])
        label = classes[cls_id]
        confidence = float(box.conf[0])
        detected_data.append({"Object": label, "Confidence": f"{confidence:.2f}"})

    return annotated_img, detected_data


def convert_to_h264(input_path, output_h264_path):
    os.system(
        f"ffmpeg -y -i {input_path} -vcodec libx264 -pix_fmt yuv420p {output_h264_path}"
    )

def detect_and_annotate_video(uploaded_file):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.flush()
    tfile.close()
    input_path = tfile.name

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs("output", exist_ok=True)
    raw_output_path = os.path.join("output", "raw_annotated.mp4")
    final_output_path = os.path.join("output", "annotated_video_final.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()
    os.remove(input_path)

    # 🔄 Convert to streamable format using ffmpeg
    ffmpeg_cmd = f'ffmpeg -y -i "{raw_output_path}" -vcodec libx264 -pix_fmt yuv420p "{final_output_path}"'
    os.system(ffmpeg_cmd)

    return final_output_path


def run_live_detection():
    stframe = st.empty()

    # Start capturing
    cap = cv2.VideoCapture(0)

    # Loop to continuously show frames
    while st.session_state.live:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera not accessible.")
            break

        results = model.predict(source=frame, conf=0.25)
        annotated = results[0].plot()

        # Show the frame
        stframe.image(annotated, channels="BGR")

    cap.release()
    # cv2.destroyAllWindows()

# === UI Tabs ===
tabs = st.tabs(["📷 Annotate Image", "🎥 Annotate Video", "📡 Live Camera"])

with tabs[0]:
    st.header("📷 Upload an Image for Annotation")
    uploaded_img = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        result_img, detected_data = detect_and_annotate_image(uploaded_img)
        st.image(result_img, caption="Annotated Image", use_container_width=True)

        if detected_data:
            st.subheader("📋 Detected Objects")
            st.table(detected_data)
        else:
            st.info("No objects detected.")


with tabs[1]:
    st.header("🎥 Upload a Video for Annotation")
    uploaded_vid = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])
    if uploaded_vid:
        st.info("Processing video. This may take a few seconds...")
        output_path = detect_and_annotate_video(uploaded_vid)

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

            st.caption(f"Video loaded successfully! Size: {len(video_bytes) / 1_000_000:.2f} MB")  # Show size

with tabs[2]:
    st.header("📡 Live Annotation from Camera")

    # Init session state if not already
    if "live" not in st.session_state:
        st.session_state.live = False

    # Start / Stop buttons
    col1, col2 = st.columns(2)
    if col1.button("▶️ Start Live Camera"):
        st.session_state.live = True
        run_live_detection()

    if col2.button("⏹️ Stop Live Camera"):
        st.session_state.live = False

