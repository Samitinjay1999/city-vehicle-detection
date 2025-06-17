from PIL import Image
import numpy as np

def detect_and_annotate_image(model, uploaded_image):
    """
    Detect and annotate objects in an uploaded image using a YOLO model.

    This function takes an uploaded image and a YOLO model, performs object detection,
    and returns the annotated image along with detection data.

    Args:
        model: A trained YOLO model for object detection.
        uploaded_image: An uploaded image file to be processed.

    Returns:
        tuple: A tuple containing:
            - annotated_img (numpy.ndarray): The image with detection annotations.
            - detected_data (list): A list of dictionaries with detection information.
            - original_image (PIL.Image): The original PIL image.
    """
    # Open the uploaded image and convert to RGB format
    image = Image.open(uploaded_image).convert("RGB")

    # Convert the PIL image to a numpy array for processing
    image_np = np.array(image)

    # Perform object detection using the YOLO model
    # conf=0.25 sets the confidence threshold for detections
    results = model.predict(source=image_np, conf=0.25)

    # Generate an annotated image with bounding boxes and labels
    annotated_img = results[0].plot()

    # Extract detection information
    detections = results[0].boxes  # Bounding box information
    classes = results[0].names     # Class names

    # Prepare detection data for display
    detected_data = []
    for box in detections:
        # Extract class ID and get the corresponding label
        cls_id = int(box.cls[0])
        label = classes[cls_id]

        # Extract confidence score
        confidence = float(box.conf[0])

        # Append detection information to the list
        detected_data.append({
            "Object": label,
            "Confidence": f"{confidence:.2f}"  # Format confidence to 2 decimal places
        })

    # Return the annotated image, detection data, and original image
    return annotated_img, detected_data, image
