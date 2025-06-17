import tempfile
import os
import cv2

def convert_to_h264(input_path, output_h264_path):
    """
    Convert a video file to H.264 codec using ffmpeg.

    Args:
        input_path (str): Path to the input video file.
        output_h264_path (str): Path where the converted H.264 video will be saved.
    """
    # Construct and execute the ffmpeg command to convert the video to H.264
    os.system(
        f"ffmpeg -y -i {input_path} -vcodec libx264 -pix_fmt yuv420p {output_h264_path}"
    )

def detect_and_annotate_video(model, uploaded_file):
    """
    Detect objects in a video file using a given model and annotate the video with the results.

    Args:
        model: The pre-trained model used for object detection.
        uploaded_file: The video file uploaded by the user.

    Returns:
        str: Path to the final annotated video file.
    """
    # Create a temporary file to store the uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.flush()
    tfile.close()
    input_path = tfile.name

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(input_path)

    # Get video properties: width, height, and frames per second (fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create an output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Define paths for the raw and final annotated videos
    raw_output_path = os.path.join("output", "raw_annotated.mp4")
    final_output_path = os.path.join("output", "annotated_video_final.mp4")

    # Initialize VideoWriter to save the annotated video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

    # Process each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Use the model to predict objects in the current frame
        results = model.predict(frame)

        # Annotate the frame with the prediction results
        annotated_frame = results[0].plot()

        # Write the annotated frame to the output video
        out.write(annotated_frame)

    # Release the video capture and writer resources
    cap.release()
    out.release()

    # Remove the temporary input file
    os.remove(input_path)

    # Convert the raw annotated video to a streamable H.264 format using ffmpeg
    ffmpeg_cmd = f'ffmpeg -y -i "{raw_output_path}" -vcodec libx264 -pix_fmt yuv420p "{final_output_path}"'
    os.system(ffmpeg_cmd)

    # Return the path to the final annotated video
    return final_output_path
