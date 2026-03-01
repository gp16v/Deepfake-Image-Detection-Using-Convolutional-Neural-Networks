# ============================================================
# app.py  — Streamlit Deepfake Detector
# Place this file in your VS Code project folder
# Run with: streamlit run app.py
# ============================================================
# Folder structure:
#  my_project/
#  ├── app.py
#  └── finetuned_final.keras
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import tempfile
import os

# ────────────────────────────────────────────────────────────
IMAGE_SIZE = 224
# ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="wide"
)

# ── Load CNN model ───────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "finetuned_final.keras"
    if not os.path.exists(model_path):
        st.error("❌ 'finetuned_final.keras' not found. Put it in the same folder as app.py")
        st.stop()
    return tf.keras.models.load_model(model_path)

model = load_model()

# ── Load OpenCV face detector ────────────────────────────────
# Uses Haar Cascade — built into OpenCV, no extra files needed
@st.cache_resource
def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        st.error("❌ Face detector failed to load.")
        st.stop()
    return detector

face_detector = load_face_detector()


# ══════════════════════════════════════════════════════════════
# DETECTION LOGIC
# ══════════════════════════════════════════════════════════════

def detect_and_crop_face(pil_image):
    """
    Detects face in a PIL image and returns the cropped face.

    Why this is needed:
      The model was trained on tightly cropped face images.
      If we feed a full photo (body, background etc.) directly,
      the model sees mostly non-face pixels and gives wrong results.
      So we detect the face region first, crop it, then pass to model.

    Returns:
      face_pil    : PIL Image — cropped face (or full image if no face found)
      face_found  : bool
      face_coords : (x, y, w, h) bounding box, or None
    """
    # Convert PIL to numpy BGR for OpenCV
    img_np  = np.array(pil_image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Run face detection
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor  = 1.1,   # how much image shrinks each pass
        minNeighbors = 5,      # higher = fewer false positives
        minSize      = (60, 60)
    )

    if len(faces) == 0:
        # No face found — return full image, model will still try
        return pil_image, False, None

    # If multiple faces, take the largest (most prominent face)
    if len(faces) > 1:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    x, y, w, h = faces[0]

    # Add 20% padding so model sees full face including hairline and chin
    pad  = int(max(w, h) * 0.20)
    ih, iw = img_np.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(iw, x + w + pad)
    y2 = min(ih, y + h + pad)

    face_crop = img_np[y1:y2, x1:x2]
    return Image.fromarray(face_crop), True, (x, y, w, h)


def draw_face_box(pil_image, face_coords, label):
    """
    Draws colored bounding box + label on the original image.
    Green = REAL, Red = FAKE
    """
    img_np = np.array(pil_image.convert('RGB')).copy()
    x, y, w, h = face_coords
    color = (0, 200, 0) if label == 'REAL' else (220, 0, 0)

    cv2.rectangle(img_np, (x, y), (x + w, y + h), color, thickness=3)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, w / 200)
    text_size  = cv2.getTextSize(label, font, font_scale, 2)[0]

    # Background fill behind text for readability
    cv2.rectangle(img_np,
                  (x, y - text_size[1] - 10),
                  (x + text_size[0] + 8, y),
                  color, -1)
    cv2.putText(img_np, label, (x + 4, y - 5),
                font, font_scale, (255, 255, 255), 2)

    return Image.fromarray(img_np)


def preprocess_for_model(pil_image):
    """
    Resize and normalize — identical to training preprocessing.
    """
    img = pil_image.convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # → (1, 224, 224, 3)


def run_detection(pil_image):
    """
    Full pipeline for one image:
      detect face → crop → preprocess → model → result

    Returns:
      label         : 'REAL' or 'FAKE'
      confidence    : 0.0 – 100.0
      face_found    : bool
      annotated_img : original with bounding box
      face_crop     : what was fed to the model
    """
    face_crop, face_found, coords = detect_and_crop_face(pil_image)
    arr        = preprocess_for_model(face_crop)
    pred       = model.predict(arr, verbose=0)
    pred_class = int(np.argmax(pred[0]))
    confidence = float(pred[0][pred_class]) * 100
    label      = 'REAL' if pred_class == 1 else 'FAKE'

    if face_found:
        annotated = draw_face_box(pil_image, coords, label)
    else:
        annotated = pil_image   # nothing to draw

    return label, confidence, face_found, annotated, face_crop


def run_detection_frame(bgr_frame):
    """
    Same pipeline for a single video frame (OpenCV BGR format).
    Returns class index, confidence, face_found bool.
    """
    rgb   = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil_f = Image.fromarray(rgb)
    face_crop, face_found, _ = detect_and_crop_face(pil_f)
    arr   = preprocess_for_model(face_crop)
    pred  = model.predict(arr, verbose=0)
    cls   = int(np.argmax(pred[0]))
    conf  = float(pred[0][cls])
    return cls, conf, face_found


# ══════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════

st.title("🔍 Deepfake Detection System")
st.markdown("*CNN-based detection using InceptionResNetV2 · Accuracy: 98.95%*")
st.markdown("---")

st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
**Project:** Deepfake Image & Video Detection  
**Model:** InceptionResNetV2  
**Dataset:** 140k Real & Fake Faces  
**Face Detector:** OpenCV Haar Cascade  

**Detection Pipeline:**  
`Upload → Detect Face → Crop Face → CNN Model → Result`

**Why face detection?**  
Model was trained on cropped faces.
Detecting & cropping before prediction gives accurate results.

**Class labels:**  
- `0` → Fake  
- `1` → Real  
""")


tab1, tab2 = st.tabs(["📷 Image Detection", "🎥 Video Detection"])


# ─── TAB 1: IMAGE ────────────────────────────────────────────
with tab1:
    st.header("Upload an Image")
    st.caption("Works best with a clear face photo. Full body or group photos are also supported.")

    uploaded = st.file_uploader(
        "Supported formats: jpg, jpeg, png",
        type=['jpg', 'jpeg', 'png'],
        key='image_uploader'
    )

    if uploaded:
        pil_img = Image.open(uploaded).convert('RGB')

        with st.spinner("Detecting face and analyzing..."):
            label, confidence, face_found, annotated_img, face_crop = run_detection(pil_img)

        # Three columns: annotated original | face crop | result
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.subheader("Detected Face")
            st.image(annotated_img, use_container_width=True)
            if face_found:
                st.caption("✅ Face detected — box shows what was analyzed")
            else:
                st.caption("⚠️ No face detected — full image was analyzed")

        with col2:
            st.subheader("What Model Saw")
            st.caption("Cropped & resized to 224×224")
            st.image(face_crop, use_container_width=True)

        with col3:
            st.subheader("Result")
            if label == 'FAKE':
                st.error(f"## ⚠️ {label}")
            else:
                st.success(f"## ✅ {label}")

            st.metric("Confidence", f"{confidence:.2f}%")
            st.progress(int(confidence))

            if not face_found:
                st.warning("No face was detected.\nPrediction may be less reliable.")


# ─── TAB 2: VIDEO ────────────────────────────────────────────
with tab2:
    st.header("Upload a Video")
    st.caption("Samples frames across the video, detects face in each frame, then classifies.")

    uploaded_video = st.file_uploader(
        "Supported formats: mp4, avi, mov",
        type=['mp4', 'avi', 'mov'],
        key='video_uploader'
    )

    if uploaded_video:
        video_bytes = uploaded_video.read()
        st.video(video_bytes)

        if st.button("🔍 Analyze Video"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            try:
                tfile.write(video_bytes)
                tfile.flush()
                tfile.close()

                with st.spinner("Extracting frames, detecting faces, analyzing..."):
                    cap      = cv2.VideoCapture(tfile.name)
                    total_f  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    interval = max(1, total_f // 60)   # ~60 frames across video

                    predictions   = []
                    no_face_count = 0
                    frame_idx     = 0
                    progress_bar  = st.progress(0.0)

                    try:
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break

                            if frame_idx % interval == 0:
                                cls, conf, face_found = run_detection_frame(frame)
                                predictions.append({'class': cls, 'conf': conf})
                                if not face_found:
                                    no_face_count += 1
                                progress_bar.progress(min(len(predictions) / 60, 1.0))

                            frame_idx += 1
                    finally:
                        cap.release()   # always released even if error mid-loop

            finally:
                if os.path.exists(tfile.name):
                    os.unlink(tfile.name)

            if not predictions:
                st.error("Could not extract frames from this video.")
            else:
                fake_n     = sum(1 for p in predictions if p['class'] == 0)
                real_n     = sum(1 for p in predictions if p['class'] == 1)
                total      = len(predictions)
                overall    = 'FAKE' if fake_n > real_n else 'REAL'
                overall_cf = max(fake_n, real_n) / total * 100

                st.subheader("Results")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Frames Analyzed", total)
                c2.metric("Fake Frames",     fake_n)
                c3.metric("Real Frames",     real_n)
                c4.metric("No Face Found",   no_face_count)

                st.markdown("---")
                if overall == 'FAKE':
                    st.error(f"## ⚠️ Video is: {overall}")
                else:
                    st.success(f"## ✅ Video is: {overall}")

                st.metric("Confidence", f"{overall_cf:.2f}%")
                st.progress(int(overall_cf))

                if no_face_count > total * 0.5:
                    st.warning(
                        f"Face not detected in {no_face_count}/{total} frames. "
                        "Video may not contain clear face footage — results may be unreliable."
                    )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:grey;'>"
    "Group 16 · S6 CSE Batch C · KTU Mini Project"
    "</div>",
    unsafe_allow_html=True
)
