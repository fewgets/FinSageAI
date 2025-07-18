# 🧠 NeuroAssist-AI

> A Deep Learning-based Smart System for Brain Tumor Detection and Glioma Stage Prediction
> 
> 🔗 **[Live Demo](https://neuroassistai.vercel.app/)**

---

## 📌 Overview

**NeuroAssist-AI** is a two-stage deep learning pipeline that automates:

1. **Brain Tumor Detection** from grayscale MRI/CT images using a custom CNN.
2. **Glioma Stage Prediction** using numerical gene mutation data via a custom ANN.

> This system is inspired by real clinical practices and aims to provide intelligent support to radiologists and neurologists.

---

## 🧪 Live Testing

You can **test the full system online**:

🌐 **[🔗 Live Web App – NeuroAssist-AI](https://neuroassistai.vercel.app/)**

---

## 📚 Research Basis

📄 Inspired by:
**“Brain Tumor Classification and Glioma Stage Prediction Using Deep Learning”**

🔗 [Read the original paper](https://onlinelibrary.wiley.com/doi/full/10.1155/2022/1830010)

> Note: Original paper had no public dataset or code - we implemented it from scratch.

---

## 📂 Dataset

Dataset: [Brain Tumor MRI Dataset – Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

* **4 Classes:** Glioma, Meningioma, Pituitary, No Tumor
* **Format:** Grayscale `.jpg` images categorized in folders

---

## 🧠 Model Architecture

### 🔷 CNN – Brain Tumor Detection

| Layer Type  | Description                                                 |
| ----------- | ----------------------------------------------------------- |
| Input       | Grayscale MRI/CT image                                      |
| Conv Blocks | 3 × Conv2D + ReLU + MaxPooling                              |
| FC Layers   | Flatten → Dense → Softmax                                   |
| Output      | 4 classes (`No Tumor`, `Meningioma`, `Pituitary`, `Glioma`) |

*✅ Trained from scratch in PyTorch*
*❌ No dropout (no overfitting observed)*

---

### 🟢 ANN – Glioma Stage Classification

| Layer Type | Description                      |
| ---------- | -------------------------------- |
| Input      | Gene mutation test results       |
| Dense      | 2–3 Fully Connected Layers       |
| Activation | ReLU + Softmax/Regression Output |
| Output     | Glioma Stage (I–IV)              |

---

## 💾 Model Files

| File Name           | Purpose                           | Availability                                                                                                           |
| ------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `BTD_model.pth`     | Brain Tumor Detection (CNN)       | 🔗 [Download from Google Drive](https://drive.google.com/file/d/19SVLCD3DTa1aBZ9PI4TTgNkvJKgL2LSY/view?usp=drive_link) |
| `glioma_stages.pth` | Glioma Stage Classification (ANN) | ✅ Included in `models/` directory                                                                                      |

📍 **Note:**
Due to GitHub’s 100MB limit, `BTD_model.pth` is stored externally.
After downloading, **manually place it inside the `models/` folder** like this:

```bash
models/BTD_model.pth
```

---

## 🔄 Optional: Auto-Download Script

Use this code to download the CNN model automatically if missing:

```python
import os, urllib.request

model_url = "https://drive.google.com/uc?export=download&id=19SVLCD3DTa1aBZ9PI4TTgNkvJKgL2LSY"
model_path = "models/BTD_model.pth"

if not os.path.exists(model_path):
    os.makedirs("models", exist_ok=True)
    print("Downloading model...")
    urllib.request.urlretrieve(model_url, model_path)
    print("Model downloaded.")
```

---

## ⚙️ Tech Stack

| Category     | Tools / Libraries                       |
| ------------ | --------------------------------------- |
| Language     | Python 3.10+                            |
| DL Framework | PyTorch                                 |
| Others       | OpenCV, NumPy, scikit-learn, Matplotlib |
| Training Env | Jupyter Notebook, NVIDIA GPU            |
| Deployment   | FastAPI + Vercel (Frontend)             |
| Storage      | Google Drive (for model hosting)        |

---

## 📁 Folder Structure

```bash
NeuroAssistAI/
├── main.py                  # Entry point
├── API.py                   # FastAPI backend
├── utils.py                 # Helper functions
├── models/
│   ├── BTD_model.pth
│   └── glioma_stages.pth
├── images/
├── README.md
└── .gitignore
```

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repo

```bash
git clone https://github.com/fewgets/NeuroAssistAI.git
cd NeuroAssistAI
```

### 2️⃣ Install Dependencies

```bash
pip install torch torchvision opencv-python matplotlib scikit-learn fastapi uvicorn
```

Or use:

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App

```bash
python main.py
```

---

## ✨ Features

✅ Tumor classification using CNN

✅ Glioma stage prediction using ANN

✅ Web interface for real-time inference

✅ Pre-trained models included

✅ End-to-end modular system

✅ Easy to extend and deploy

---

## 🔭 Future Enhancements

* 🤖 Chatbot integration for medical Q\&A (Gemini/GPT)
* 🧬 Integration with real-time genetic APIs
* 📊 Real-time dashboard for doctors
* 📱 Mobile version using React Native

---

## 📩 Contact

**👨‍💻 Usama Shahid**
📧 Email: [dev.usamashahid@gmail.com](mailto:dev.usamashahid@gmail.com)
🔗 GitHub: [@fewgets](https://github.com/fewgets)

> For training notebooks, collab requests, or guidance — feel free to connect.

---

## ⚠️ License

This project is intended for **academic and educational purposes only.**
Re-use is allowed with proper credit and citation.
Not intended for clinical deployment without validation.

---

### 🔗 [🌐 Live App](https://neuroassistai.vercel.app/) • [📁 Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) • [📜 Paper](https://onlinelibrary.wiley.com/doi/full/10.1155/2022/1830010)

---
