import sys
import os
import time
from ultralytics import YOLO

# --- Configuration ---
PROJECT_NAME = "glassesyolov8n_50e"
DATA_YAML = "data.yaml"
IMG_SIZE = 640
EPOCHS = 50
BATCH_SIZE = 16
WORKERS = 4
PATIENCE = 10
DEVICE = 0


try:
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    from rich.table import Table
    from rich.console import Console
    from rich import box

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich library not found. Running in standard mode.")


class TrainingDashboard:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.console = Console()
        self.layout = Layout()

        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3),
        )

        self.layout["body"].split_row(
            Layout(name="progress", ratio=1), Layout(name="stats", ratio=1)
        )

        self.overall_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None, style="cyan", complete_style="deep_sky_blue1"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            expand=True,
        )
        self.epoch_task = self.overall_progress.add_task(
            "[b]Total Progress", total=total_epochs
        )

        self.stats_table = Table(box=box.SIMPLE_HEAD)
        self.stats_table.add_column("Metric", style="cyan")
        self.stats_table.add_column("Value", style="magenta")

        self.update_layout()

    def update_layout(self):
        self.layout["header"].update(
            Panel(
                "YOLOv8 Training Dashboard", style="bold white on blue", box=box.HEAVY
            )
        )

        self.layout["progress"].update(
            Panel(self.overall_progress, title="Training Status", border_style="green")
        )

        self.layout["stats"].update(
            Panel(self.stats_table, title="Current Metrics", border_style="yellow")
        )

        self.layout["footer"].update(Panel("Press Ctrl+C to stop", style="dim white"))

    def update_progress(self, epoch):
        self.overall_progress.update(self.epoch_task, completed=epoch + 1)

    def update_metrics(self, metrics):
        self.stats_table = Table(box=box.SIMPLE_HEAD)
        self.stats_table.add_column("Metric", style="cyan")
        self.stats_table.add_column("Value", style="magenta")
        for k, v in metrics.items():
            self.stats_table.add_row(k, f"{v:.4f}")
        self.layout["stats"].update(
            Panel(self.stats_table, title="Current Metrics", border_style="yellow")
        )


dashboard = None


def on_train_epoch_end(trainer):
    if dashboard:
        dashboard.update_progress(trainer.epoch)
        metrics = {
            "mAP50": trainer.metrics.get("metrics/mAP50(B)", 0),
            "mAP50-95": trainer.metrics.get("metrics/mAP50-95(B)", 0),
            "Box Loss": trainer.loss_items[0].item()
            if len(trainer.loss_items) > 0
            else 0,
            "CLS Loss": trainer.loss_items[1].item()
            if len(trainer.loss_items) > 1
            else 0,
        }
        dashboard.update_metrics(metrics)


if __name__ == "__main__":
    train_args = {
        "data": DATA_YAML,
        "imgsz": IMG_SIZE,
        "epochs": EPOCHS,
        "batch": BATCH_SIZE,
        "name": PROJECT_NAME,
        "device": DEVICE,
        "patience": PATIENCE,
        "workers": WORKERS,
        "exist_ok": True,
        "verbose": True,
        "plots": True,
    }

    model = YOLO("yolov8n.pt")

    if RICH_AVAILABLE:
        dashboard = TrainingDashboard(total_epochs=EPOCHS)

        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        print("Starting UI...")
        try:
            with Live(dashboard.layout, refresh_per_second=4, screen=True):
                model.train(**train_args)
        except Exception as e:
            print(f"UI Error: {e}")
            print("Falling back to standard mode...")
            model.train(**train_args)

    else:
        print("\n" + "=" * 50)
        print(f"Starting YOLOv8 Training: {PROJECT_NAME}")
        print("=" * 50 + "\n")

        model.train(**train_args)

    print("\n" + "=" * 50)
    print("Exporting Model to ONNX format...")
    print("=" * 50 + "\n")

    best_model_path = os.path.join("runs", "detect", PROJECT_NAME, "weights", "best.pt")

    try:
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}...")
            export_model = YOLO(best_model_path)
            success = export_model.export(format="onnx")
            print(f"Export completed: {success}")
        else:
            print(
                f"Warning: Best model not found at {best_model_path}. Exporting current model state."
            )
            success = model.export(format="onnx")
            print(f"Export completed: {success}")
    except Exception as e:
        print(f"Export functionality encountered an error: {e}")
