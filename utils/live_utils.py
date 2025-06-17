import streamlit as st
import cv2

def run_live_detection(model):
    """
    Run live object detection using a YOLO model and display the results in a Streamlit app.

    This function captures video from the default camera, performs real-time object detection
    using the provided YOLO model, and displays the annotated frames in a Streamlit app.
    The detection continues as long as the 'live' state in the session is True.

    Args:
        model: A trained YOLO model for object detection.

    Returns:
        None: This function does not return any values but displays the live detection results.
    """
    # Create an empty placeholder for the video frame in the Streamlit app
    stframe = st.empty()

    # Initialize video capture from the default camera (index 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        st.error("Could not open camera.")
        return

    # Main loop for live detection
    # Continues as long as the 'live' state in the session is True
    while st.session_state.live:
        # Read a frame from the camera
        ret, frame = cap.read()

        # If frame is not read correctly, show warning and break the loop
        if not ret:
            st.warning("Camera not accessible.")
            break

        # Perform object detection on the current frame
        # conf=0.25 sets the confidence threshold for detections
        results = model.predict(source=frame, conf=0.25)

        # Annotate the frame with detection results
        annotated = results[0].plot()

        # Display the annotated frame in the Streamlit app
        # Note: OpenCV uses BGR format by default, so we specify channels="BGR"
        stframe.image(annotated, channels="BGR")

    # Release the camera resources
    cap.release()

    # Note: cv2.destroyAllWindows() is commented out as it's not typically needed in Streamlit apps
    # cv2.destroyAllWindows()
