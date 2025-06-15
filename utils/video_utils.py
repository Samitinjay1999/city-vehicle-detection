import tempfile
import os
import cv2

def convert_to_h264(input_path, output_h264_path):
    os.system(
        f"ffmpeg -y -i {input_path} -vcodec libx264 -pix_fmt yuv420p {output_h264_path}"
    )

def detect_and_annotate_video(model,uploaded_file):
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