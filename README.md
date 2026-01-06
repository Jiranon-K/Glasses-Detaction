<div align="center">

# 👁️ Smart Object Detection System
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
-   **🎨 Modern UI**: Custom-drawn overlays, toast notifications, and non-intrusive bounding boxes.
-   **🛡️ Smart Cooldown**: Prevents notification spam with intelligent timer logic.

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

### Run the Application
Start the detection system with a single command:
```bash
python LAB09APPLY.py
```

### 🎮 Controls
| Key | Action |
| :---: | :--- |
| **`q`** | Quit the application safely |

---

## 🏗️ Project Structure
```
.
├── LAB09APPLY.py    # 🟢 Main Application Entry Point
├── train_model.py   # 🧠 Model Training Script
├── requirements.txt # 📦 Dependencies
├── .env             # 🔑 Secrets (Not committed)
└── runs/            # 📂 YOLO Generated Outputs
```

---

<div align="center">
    <sub>Designed for minimal overhead and maximum efficiency.</sub>
</div>
