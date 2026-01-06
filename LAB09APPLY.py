import cv2
import requests
import time
import os
import threading
import datetime
from dotenv import load_dotenv
from ultralytics import YOLO

# ==============================================================================
# ⚙️ SYSTEM CONFIGURATION
# ==============================================================================
# Load variables from .env file
load_dotenv()


class SmartDetector:
    def __init__(self):
        # Configuration
        self.TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
        self.MODEL_PATH = "runs/detect/glassesyolov8n_50e/weights/best.pt"
        self.CONFIDENCE_THRESHOLD = 0.5
        self.COOLDOWN_SECONDS = 60

        # UI Colors (BGR)
        self.COLOR_PRIMARY = (255, 191, 0)  # Deep Sky Blue
        self.COLOR_TEXT = (255, 255, 255)  # White
        self.COLOR_BG = (20, 20, 20)  # Dark Gray
        self.COLOR_ALERT = (0, 0, 255)  # Red
        self.COLOR_SUCCESS = (0, 255, 0)  # Green
        self.COLOR_GRAY = (100, 100, 100)  # Gray

        # State Variables
        self.last_notification_time = 0
        self.telegram_enabled = True  # Toggle state
        self.notification_queue = []
        self.running = True

        # FPS Calculation
        self.prev_frame_time = 0
        self.new_frame_time = 0

        # Initialize Model
        if not os.path.exists(self.MODEL_PATH):
            raise FileNotFoundError(f"❌ Model missing: {self.MODEL_PATH}")
        print(f"🔄 Loading model: {self.MODEL_PATH}...")
        self.model = YOLO(self.MODEL_PATH)

        # Initialize Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.cap.isOpened():
            raise RuntimeError("❌ Unknown Camera Source")

        print("✅ System Initialized")
        self.add_toast("[SYSTEM] Online. Press 'q' to quit, 't' to toggle Telegram")

    # ==============================================================================
    # 📡 NOTIFICATION SERVICE
    # ==============================================================================
    def send_telegram_photo_task(self, caption, image_path):
        """Worker function for threading"""
        if not self.TELEGRAM_TOKEN or not self.TELEGRAM_CHAT_ID:
            return

        url = f"https://api.telegram.org/bot{self.TELEGRAM_TOKEN}/sendPhoto"
        try:
            with open(image_path, "rb") as photo:
                resp = requests.post(
                    url,
                    data={"chat_id": self.TELEGRAM_CHAT_ID, "caption": caption},
                    files={"photo": photo},
                )
                if resp.status_code == 200:
                    print(f"✅ Notification Sent: {caption}")
                    self.add_toast(f"[SUCCESS] Sent: {caption}")
                else:
                    print(f"❌ Upload Failed: {resp.text}")
                    self.add_toast("[ERROR] Send Failed")
        except Exception as e:
            print(f"❌ Photo Error: {e}")
            self.add_toast("[ERROR] Connection Error")
        finally:
            # Cleanup temp file
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except:
                    pass

    def send_notification(self, message, frame):
        """Prepares and launches notification thread"""
        if not self.telegram_enabled:
            return

        # Generate unique filename using timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_file = f"snap_{timestamp}.jpg"

        # Save frame
        cv2.imwrite(temp_file, frame)

        # Start Thread
        thread = threading.Thread(
            target=self.send_telegram_photo_task, args=(message, temp_file)
        )
        thread.daemon = True
        thread.start()

    # ==============================================================================
    # 🎨 UI & VISUALIZATION
    # ==============================================================================
    def add_toast(self, message, duration=3.0):
        self.notification_queue.append(
            {
                "text": message,
                "expiry": time.time() + duration,
                "color": self.COLOR_SUCCESS
                if "[SUCCESS]" in message or "Online" in message
                else (self.COLOR_ALERT if "[ERROR]" in message else self.COLOR_PRIMARY),
            }
        )

    def draw_ui(self, img, detected_items):
        h, w = img.shape[:2]
        current_time = time.time()

        # 1. FPS Counter
        self.new_frame_time = current_time
        fps = (
            1 / (self.new_frame_time - self.prev_frame_time)
            if self.prev_frame_time > 0
            else 0
        )
        self.prev_frame_time = self.new_frame_time
        cv2.putText(
            img,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        # 2. Main Status Banner (Bottom)
        status = "SCANNING..."
        status_color = self.COLOR_GRAY

        if detected_items:
            status = f"DETECTED: {detected_items[0]}"
            status_color = self.COLOR_ALERT

        # Background for status
        cv2.rectangle(img, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(
            img, status, (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2
        )

        # 3. Telegram Toggle Status (Top Right)
        tg_text = "TG: ON" if self.telegram_enabled else "TG: OFF"
        tg_color = self.COLOR_SUCCESS if self.telegram_enabled else self.COLOR_GRAY

        # Calculate size to align right
        (tw, th), _ = cv2.getTextSize(tg_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (w - tw - 30, 15), (w - 10, 50), (40, 40, 40), -1)
        cv2.putText(
            img, tg_text, (w - tw - 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tg_color, 2
        )

        # 4. Cooldown Timer (Below TG Status)
        time_diff = current_time - self.last_notification_time
        if self.telegram_enabled and time_diff < self.COOLDOWN_SECONDS:
            remaining = int(self.COOLDOWN_SECONDS - time_diff)
            cd_text = f"Wait: {remaining}s"
            (cw, ch), _ = cv2.getTextSize(cd_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (w - cw - 30, 60), (w - 10, 90), (40, 40, 40), -1)
            cv2.putText(
                img,
                cd_text,
                (w - cw - 20, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        # 5. Toast Notifications (Left Side)
        self.notification_queue = [
            n for n in self.notification_queue if n["expiry"] > current_time
        ]
        y_start = 80
        for note in self.notification_queue:
            text = note["text"]
            color = note.get("color", self.COLOR_TEXT)

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(
                img, (15, y_start - 25), (30 + tw, y_start + 10), (0, 0, 0), -1
            )
            cv2.putText(
                img,
                text,
                (25, y_start),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                1,
                cv2.LINE_AA,
            )
            y_start += 40

    def draw_corners(self, frame, x1, y1, x2, y2, color, thickness=2, length=20):
        # Top-Left
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
        # Top-Right
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
        # Bottom-Left
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
        # Bottom-Right
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)

    def draw_detections(self, frame, results):
        detected_names = []
        for r in results:
            for box in r.boxes:
                if box.conf[0] > self.CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    name = self.model.names[cls_id]
                    detected_names.append(name)

                    # Draw Corners (instead of full box)
                    self.draw_corners(
                        frame,
                        x1,
                        y1,
                        x2,
                        y2,
                        self.COLOR_PRIMARY,
                        thickness=3,
                        length=30,
                    )

                    # Draw Label (at bottom, minimal style)
                    label = f"{name.upper()} {conf:.0%}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        self.COLOR_TEXT,
                        1,
                        cv2.LINE_AA,
                    )
        return detected_names

    # ==============================================================================
    # 🚀 RUN LOOP
    # ==============================================================================
    def run(self):
        window_name = "Smart Detection System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        try:
            while self.running:
                success, frame = self.cap.read()
                if not success:
                    break

                # --- Inference ---
                results = self.model(frame, verbose=False)
                detected_items = self.draw_detections(frame, results)

                # --- Logic ---
                current_time = time.time()
                if detected_items and self.telegram_enabled:
                    # Check Cooldown
                    if (
                        current_time - self.last_notification_time
                        > self.COOLDOWN_SECONDS
                    ):
                        target = detected_items[0]
                        msg = f"Detected: {target}"

                        self.add_toast(f"[ALERT] Sending photo...", duration=5.0)
                        self.send_notification(msg, frame)
                        self.last_notification_time = current_time

                # --- Draw Interface ---
                self.draw_ui(frame, detected_items)

                cv2.imshow(window_name, frame)

                # --- Controls ---
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("t"):
                    self.telegram_enabled = not self.telegram_enabled
                    state = "ON" if self.telegram_enabled else "OFF"
                    self.add_toast(f"[SYSTEM] Telegram: {state}")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("🛑 System Shutdown")


if __name__ == "__main__":
    try:
        app = SmartDetector()
        app.run()
    except Exception as e:
        print(f"❌ Critical Error: {e}")
