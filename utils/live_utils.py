import streamlit as st
import cv2

def run_live_detection(model):
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