from PIL import Image
import numpy as np
# === Helper function ===
def detect_and_annotate_image(model,uploaded_image):
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