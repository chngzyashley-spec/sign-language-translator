# sign-language-translator
Real-time SgSl translator powered by MediaPipe and TensorFlow, capable of recognising both static hand poses and dynamic sign actions through computer vision.

---

## 📌 Overview

This project aims to provide a trainable and extensible translator for **Singapore Sign Language (SgSL)** using webcam-based hand tracking and deep learning models.

The system captures hand landmarks using MediaPipe, processes the data into machine-learning-ready features, and performs real-time sign prediction using trained neural networks.

---

## 🧠 Features

- ✋ Real-time hand tracking with MediaPipe
- 📷 Webcam-based sign detection using OpenCV
- 🧩 Static sign recognition using Feedforward Neural Networks (FNN)
- 🎬 Dynamic action recognition using sequence models (LSTM/GRU)
- ⚡ Live prediction overlay with confidence filtering
- 🛠 Trainable on custom SgSL datasets

---

## 🏗 Architecture

### 1. Data Acquisition
- **OpenCV** → Webcam access
- **MediaPipe Hands** → Hand landmark extraction

### 2. Data Processing
- Landmark normalization
- Coordinate flattening
- Sequence generation for dynamic actions

### 3. Machine Learning Models

#### Static Signs
Feedforward Neural Network (FNN)
- Detects hand poses such as:
  - A
  - B
  - Hello
  - Numbers

#### Dynamic Signs
LSTM / GRU Sequence Models
- Detects motion-based signs such as:
  - Thank You
  - Goodbye
  - Greetings

---

## 🔄 Workflow

### 1️⃣ Collect Training Data

Run:

```bash
python collect_data.py
```

This captures hand landmarks and stores them into CSV datasets for training.

---

### 2️⃣ Train the Model

Run:

```bash
python train_model.py
```

This generates:
- `model_static.h5`
- `model_dynamic.h5`
- Label encoders

---

### 3️⃣ Real-Time Translation

Run:

```bash
python translate.py
```

The application will:
- Open your webcam
- Detect hand signs in real time
- Display live predictions on screen

Press `q` to quit.

---

## 📂 Project Structure

```text
.
├── collect_data.py
├── train_model.py
├── translate.py
├── utils.py
├── requirements.txt
├── model_static.h5
├── model_dynamic.h5
├── encoder_static.pkl
├── encoder_dynamic.pkl
└── datasets/
```

---

## ⚙️ Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🖥 Technologies Used

- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy
- Computer Vision
- Deep Learning

---

## 🎯 Detecting Dynamic Signs

The current system supports both:
- **Static hand poses**
- **Motion-based sign actions**

Dynamic recognition works by:
1. Recording sequences of frames
2. Using temporal landmark data
3. Feeding sequences into LSTM/GRU models
4. Predicting actions using sliding windows
