# Deepfake Image Detection Using Convolutional Neural Networks

## 📌 Project Overview

Deepfake technology uses artificial intelligence to manipulate images and videos, creating realistic but fake visual content. With the rapid growth of social media and digital communication, detecting deepfake content has become critical for preventing misinformation, identity fraud, and digital manipulation.

This project presents a deep learning–based system that detects deepfake images using Convolutional Neural Networks (CNN). The model leverages transfer learning with the InceptionResNetV2 architecture and is enhanced using compression-aware augmentation techniques to improve robustness against image degradation.

The system classifies images into:

- Real  
- Fake  

A Streamlit-based web application allows users to upload images and receive predictions with confidence scores.

---

## 🧠 Model Details

- **Architecture:** InceptionResNetV2 (Transfer Learning)
- **Task:** Binary Image Classification (Real vs Fake)
- **Input Size:** 224 × 224
- **Output Layer:** Softmax (2 Classes)
- **Preprocessing:** Rescaling (1./255)
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy

---

## 📊 Dataset

- **Test Set Size:** 20,000 Images  
- **Distribution:**  
  - 10,000 Real  
  - 10,000 Fake  
- Balanced dataset for unbiased evaluation.
- Images resized and normalized before training and testing.

---

## 📈 Model Performance

**Decision Threshold:** 0.5  

- **Accuracy:** 98.95%  

The model demonstrates excellent discrimination capability between real and fake images and strong generalization performance on unseen data.

---


---

## 🖥 Application

The project includes a Streamlit web interface.

### Features:
- Upload an image  
- Real-time prediction  
- Confidence score display  
- Simple and clean UI  

Run the app locally to test deepfake detection interactively.

---

## 🛠 Technologies Used

**Frontend:**  
- Streamlit  

**Backend & Model Development:**  
- Python  
- TensorFlow  
- Keras  
- OpenCV  
- Albumentations  
- NumPy  
- Matplotlib  

**Model Architecture:**  
- InceptionResNetV2  

---

## 📂 Project Structure

```
deepfake_project/
│
├── app.py
├── best_model.keras
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙ Installation Guide

### 1️⃣ Clone the Repository
```
git clone <your-repository-link>
cd deepfake_project
```

### 2️⃣ Create Virtual Environment
```
python -m venv venv
```

### 3️⃣ Activate Virtual Environment

Windows:
```
venv\Scripts\activate
```

Mac/Linux:
```
source venv/bin/activate
```

### 4️⃣ Install Dependencies
```
pip install -r requirements.txt
```

### 5️⃣ Run the Application
```
streamlit run app.py
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

155	RASIN JOHN
154	RAHUL M J
151	NOEL REJI
165	SATH SREE HARI K
