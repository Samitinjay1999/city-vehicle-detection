
# 🚦 Real-Time Object Detection System using YOLOv8

This project is a real-time object detection system that can annotate images, videos, and live webcam feeds using a YOLOv8 model. It supports object classes: **Car**, **Two-wheeler**, and **Autorickshaw**. The project is built using **Streamlit** for the UI and **Ultralytics YOLOv8** for object detection.

---

## 📋 Features

- 🎯 **Image Annotation**  
  Upload an image and get annotated output with bounding boxes and confidence scores displayed in a table.

- 🎬 **Video Annotation**  
  Upload a video, annotate every frame using the YOLOv8 model, and preview the result directly in the app.

- 📡 **Live Camera Annotation**  
  Use your device's webcam to detect and annotate objects in real-time.

---

## 📁 File Structure

```
Object-Detection-System/
│
├── model/
│   └── best.pt                  # Trained YOLOv8 model with 3 classes
│
├── output/
│   └── annotated_video.mp4      # Processed video output
│
├── app.py                       # Main Streamlit application
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
```

---

## ✅ Requirements

- Python 3.8+
- Streamlit
- OpenCV
- Ultralytics YOLOv8
- ffmpeg (installed and accessible via PATH)

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Run the App Locally

```bash
streamlit run app.py
```

Make sure your webcam is accessible and your `best.pt` model file is placed inside the `model/` directory.

---

## 🌐 Deployment

You can deploy this app on:
- **Streamlit Cloud** (requires webcam permissions)
- **Local Network** (to access from your phone/laptop)

### ⚠️ Live Camera Notes:
- Live camera will work **only on the device** where the app is running (due to browser security limitations).
- On mobile, if you deploy via local network, the mobile browser must support webcam access.

---

## 📊 Output Example

### Image Tab:
- Annotated image preview
- Table of detected objects with confidence values

| Object       | Confidence |
|--------------|------------|
| car          | 0.87       |
| two_wheeler  | 0.79       |

### Video Tab:
- Processed video preview in a compact video player
- Supports `.mp4`, `.avi`, `.mov`

---

## 🧠 YOLOv8 Model Info

- Trained using annotated data from CVAT
- Format: YOLOv8 (compatible with Ultralytics)
- Classes:
  - `car`
  - `two_wheeler`
  - `autorickshaw`

---

<!-- ## 📸 Screenshots

### 1. Image Annotation  
![Image Annotation Example](screenshots/image_example.jpg)

### 2. Video Annotation  
![Video Annotation Example](screenshots/video_example.jpg)

### 3. Live Detection  
![Live Detection Example](screenshots/live_example.jpg)

--- -->

## 🛠 Future Improvements

- Add bounding box coordinates to the table
- model improvement for all type of vehicle 
---

## ✍️ Scope Covered

✔ Image upload + annotation  
✔ Video upload + annotation + inline preview  
✔ Live camera real-time detection  
✔ Display detected class + confidence  
✔ Streamlit UI with tabs for clear separation  
✔ Compact embedded video player  
✔ Model trained and integrated successfully

---

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://docs.ultralytics.com)
- [Streamlit](https://streamlit.io)
- [OpenCV](https://opencv.org)
