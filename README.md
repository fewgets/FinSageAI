# 🧠 Brain Tumor Classification & Glioma Stage Detection

> A deep learning-based system for brain tumor detection from medical scans and Glioma stage prediction using gene mutation input.

---

## 📌 Project Summary

This project implements a two-stage smart classification pipeline:

1. **Brain Tumor Classification** using a **custom CNN**:

   * 📅 Input: MRI/CT grayscale brain image
   * 🔂 Output: `No Tumor`, `Meningioma`, `Pituitary`, or `Glioma`

2. **Glioma Stage Classification** using a **custom ANN** (if Glioma is detected):

   * 📅 Input: Gene mutation test results (numerical values)
   * 🔂 Output: Predicted Glioma stage (e.g., Stage I–IV)

---

## 📚 Paper Reference

📄 This project is inspired by a published research paper:

🔗 [Click here to read the original paper](https://onlinelibrary.wiley.com/doi/full/10.1155/2022/1830010)

> Note: The original research did **not include dataset or code**. We implemented the full system from scratch, including dataset sourcing and model building.

---

## 📂 Dataset Used

We used a publicly available dataset of brain MRI images from Kaggle:

🔗 [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

* Total 4 classes: `Glioma`, `Meningioma`, `Pituitary`, `No Tumor`
* Images are grayscale `.jpg` format and categorized into folders

---


## 🧠 Model Architecture

### 🔵 CNN – Tumor Type Classification

* ✅ Input: Brain MRI image
* ✅ 3 Convolutional layers (Conv2D + ReLU + MaxPooling)
* ✅ Flatten + Fully connected layers
* ❌ No dropout (no overfitting observed)
* ✅ Output: 4-class softmax

### 🟩 ANN – Glioma Stage Classification

* ✅ Input: Gene mutation test results (numerical features)
* ✅ 2–3 Dense layers with ReLU activation
* ✅ Output: Stage prediction (multi-class or regression-based)

---

## 💾 Saved Models

This repository includes **pre-trained models** for direct inference:

| Model | Purpose                     | File Name                         |
| ----- | --------------------------- | --------------------------------- |
| CNN   | Brain Tumor Classification  | `models/BTD_model.pth`            |
| ANN   | Glioma Stage Classification | `models/glioma_stages.pth`        |

📆 Both models were **trained from scratch** using PyTorch.

⚠️ The models are for **inference only**.
📩 **For training code**, please DM or email the author (contact below).

---

## 🫠 Tech Stack

| Category     | Tools / Libraries                           |
| ------------ | ------------------------------------------- |
| Language     | Python 3.10+                                |
| Framework    | PyTorch                                     |
| Other Libs   | OpenCV, NumPy, Matplotlib, scikit-learn     |
| Training Env | Jupyter Notebook + NVIDIA GPU               |
| Storage      | Google Drive (for large `.pth` model files) |
| Version Ctrl | Git, GitHub                                 |

---

## 📁 Folder Structure

```bash
BrainTumorClassification/
├── main.py                     # Entry point: run this
├── ann_model.py                # ANN logic for glioma stage
├── cnn_model.py                # CNN architecture for image classification
├── utils/                      # Helper functions (preprocessing, loading, etc.)
├── dataset/                    # Image data (if any sample added)
├── models/
│   ├── BTD_model.pth           # (Download from Drive and place the file here)
│   ├── glioma_stages.pth           # Saved Model for Glioma Stages Detection
│   ├── BrainTumorClassification.ipynb   # Notebook for CNN testing
│   └── Glioma_Stages.ipynb               # Notebook for ANN training/testing
├── README.md
└── .gitignore
```

---

## 🔗 Model Download (BTD\_model.pth)

Due to GitHub's 100MB limit, the trained CNN model is stored externally.

📅 [Click here to download BTD\_model.pth](https://drive.google.com/file/d/19SVLCD3DTa1aBZ9PI4TTgNkvJKgL2LSY/view?usp=drive_link)

After downloading, place it inside the `models/` folder:

```bash
models/BTD_model.pth
```

---

## 🔄 Auto-Download Script (Optional)

Add this to `main.py` or any script to download the model automatically if it's missing:

```python
import os
import urllib.request

model_url = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
model_path = "models/BTD_model.pth"

if not os.path.exists(model_path):
    os.makedirs("models", exist_ok=True)
    print("Downloading model...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Model downloaded.")
```

---

## ⚙️ How to Run the Project

### 🔧 1. Clone the Repository

```bash
git clone https://github.com/your-username/BrainTumorClassification.git
cd BrainTumorClassification
```

### 🔧 2. Install Requirements

```bash
pip install torch torchvision opencv-python matplotlib scikit-learn
```

Or if you have `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 🚀 3. Run the App

```bash
python main.py
```

* Upload a grayscale brain scan image
* Get tumor prediction
* If **Glioma**, provide further gene test input
* ANN predicts Glioma stage

---

## ✨ Features

* ✅ End-to-end deep learning pipeline
* ✅ Accurate multi-class tumor classification
* ✅ Secondary stage detection for Glioma
* ✅ Lightweight and reproducible
* ✅ Easily extendable for new tumor types

---

## 🧠 Future Work

* Add Gemini AI chatbot integration for medical advice
* Use transfer learning (VGG16, ResNet) for comparison
* Deploy as a web app or API

---

## 📩 Contact

👤 **Usama Shaikh**
📧 Email: [shaikhusama541@gmail.com](mailto:shaikhusama541@gmail.com)
🔗 GitHub: [@fewgets](https://github.com/fewgets)

Feel free to reach out for:

* 🔓 Training code
* 🧪 Data processing pipeline
* 🤝 Collaboration
* 💬 Guidance

---

## 📜 License

This project is for academic and research purposes only.
Feel free to fork, reference, and learn - but give credit where due 🙏

---
