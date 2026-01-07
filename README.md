<div align="center">

# Smart Object Detection System
### Real-time YOLOv8 Inference with Telegram Notifications

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-green?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-red?style=for-the-badge&logo=opencv)

<br/>

**Minimally designed. Information rich. Seamlessly integrated.**

</div>

---

## ✨ Features

-   **⚡ Real-time Detection**: Powered by YOLOv8 for high-speed, accurate object recognition.
-   **📱 Instant Alerts**: Asynchronous Telegram notifications with snapshot images.
-   **📊 Advanced Training Dashboard**: Uses `Rich` library to display real-time training metrics, progress bars, and loss statistics in the terminal.
-   **🔄 Auto-Export**: Automatically exports the best model to **ONNX** format after training completes.
-   **📈 Visual Analytics**: Generates label correlograms (`labels_correlogram.jpg`) to visualize dataset statistics.
-   **🎨 Modern UI**: Custom-drawn overlays, toast notifications (`[SUCCESS]`, `[ALERT]`), and non-intrusive corner-style bounding boxes.
-   **🛡️ Smart Cooldown**: Prevents notification spam with intelligent timer logic (default 60s).

---

## 🏗️ Project Structure
```
.
├── LAB09APPLY.py    # 🟢 Main Application (Real-time Detection & Telegram)
├── train_model.py   # 🧠 Model Training (with Rich Dashboard & ONNX Export)
├── requirements.txt # 📦 Project Dependencies
├── data.yaml        # 📄 Dataset Configuration
├── yolov8n.pt       # 🤖 Pre-trained YOLOv8 Model (Base)
├── yolo11n.pt       # 🤖 Pre-trained YOLOv11 Model (Alternate)
├── .env             # 🔑 Environment Variables (Secrets)
├── runs/            # 📂 Training & Inference Outputs (logs, weights, plots)
├── train/           # 📂 Training Dataset (images/labels)
├── valid/           # 📂 Validation Dataset (images/labels)
└── test/            # 📂 Testing Dataset (images/labels)
```

---

## 🛠️ Installation

### 1. Clone & Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory:
```env
TELEGRAM_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

---

## 🚀 Usage

### 1. Training the Model (`train_model.py`)
Train a custom YOLOv8 model with a beautiful terminal dashboard.
```bash
python train_model.py
```
**Key Features:**
- **Rich Dashboard**: Live progress tracking, epoch stats, and loss metrics.
- **Auto-Export**: Converts `best.pt` to `best.onnx` automatically.
- **Analytics**: Forces generation of `labels_correlogram.jpg` for data distribution analysis.

### 2. Running Inference (`LAB09APPLY.py`)
Start the real-time detection system with webcam feed and Telegram alerts.
```bash
python LAB09APPLY.py
```

### 🎮 Controls
| Key | Action |
| :---: | :--- |
| **`t`** | **Toggle Telegram** on/off (prevents sending alerts) |
| **`q`** | **Quit** the application safely |

---

<div align="center">
    <sub>Designed for minimal overhead and maximum efficiency.</sub>
</div>
