from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QSpinBox, QMessageBox, QSizePolicy
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from app.services.data_service import load_split_context, get_dataset_size
from app.services.model_service import build_and_load_model
from app.services.analysis_service import run_full_analysis
from app.plotting.time_series_renderer import render_time_series_explanation


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(12, 8))
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Time Series XAI Viewer")
        self.resize(1400, 900)

        self.context = None
        self.model = None
        self.current_split = "test"

        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        # Dataset row
        dataset_row = QHBoxLayout()
        dataset_row.addWidget(QLabel("Dataset:"))
        self.dataset_edit = QLineEdit()
        self.dataset_edit.setPlaceholderText("Path to dataset stem or *_vehicles.pcl")
        dataset_row.addWidget(self.dataset_edit)

        self.dataset_browse_btn = QPushButton("Browse")
        self.dataset_browse_btn.clicked.connect(self._browse_dataset)
        dataset_row.addWidget(self.dataset_browse_btn)

        root.addLayout(dataset_row)

        # Checkpoint row
        checkpoint_row = QHBoxLayout()
        checkpoint_row.addWidget(QLabel("Checkpoint:"))
        self.checkpoint_edit = QLineEdit()
        self.checkpoint_edit.setPlaceholderText("Path to .pt checkpoint")
        checkpoint_row.addWidget(self.checkpoint_edit)

        self.checkpoint_browse_btn = QPushButton("Browse")
        self.checkpoint_browse_btn.clicked.connect(self._browse_checkpoint)
        checkpoint_row.addWidget(self.checkpoint_browse_btn)

        root.addLayout(checkpoint_row)

        # Controls row
        controls_row = QHBoxLayout()

        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self._load_workspace)
        controls_row.addWidget(self.load_btn)

        controls_row.addWidget(QLabel("Sample idx:"))
        self.sample_spin = QSpinBox()
        self.sample_spin.setMinimum(0)
        self.sample_spin.setMaximum(0)
        self.sample_spin.setValue(0)
        controls_row.addWidget(self.sample_spin)

        self.prev_btn = QPushButton("Prev")
        self.prev_btn.clicked.connect(self._prev_sample)
        controls_row.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self._next_sample)
        controls_row.addWidget(self.next_btn)

        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self._analyze_current_sample)
        controls_row.addWidget(self.analyze_btn)

        controls_row.addStretch()
        root.addLayout(controls_row)

        # Status
        self.status_label = QLabel("Select dataset and checkpoint, then click Load.")
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        root.addWidget(self.status_label)

        # Canvas
        self.canvas = MplCanvas(self)
        root.addWidget(self.canvas, stretch=1)

    def _browse_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select dataset file",
            "",
            "Dataset files (*.pcl *.joblib);;All files (*.*)"
        )
        if path:
            self.dataset_edit.setText(path)

    def _browse_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select checkpoint",
            "",
            "PyTorch checkpoint (*.pt);;All files (*.*)"
        )
        if path:
            self.checkpoint_edit.setText(path)

    def _load_workspace(self):
        dataset_path = self.dataset_edit.text().strip()
        checkpoint_path = self.checkpoint_edit.text().strip()

        if not dataset_path:
            QMessageBox.warning(self, "Missing dataset", "Please select a dataset path.")
            return

        if not checkpoint_path:
            QMessageBox.warning(self, "Missing checkpoint", "Please select a checkpoint path.")
            return

        try:
            self.status_label.setText("Loading dataset...")
            self.context = load_split_context(dataset_path)

            self.status_label.setText("Loading checkpoint...")
            self.model = build_and_load_model(checkpoint_path)

            test_size = get_dataset_size(self.context, split=self.current_split)
            if test_size <= 0:
                raise ValueError(f"No samples available in split '{self.current_split}'.")

            self.sample_spin.setMaximum(test_size - 1)
            self.sample_spin.setValue(0)

            self.status_label.setText(
                f"Loaded. Split={self.current_split}, samples={test_size}, sample_idx=0"
            )

            self._analyze_current_sample()

        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))
            self.status_label.setText("Load failed.")

    def _analyze_current_sample(self):
        if self.context is None or self.model is None:
            QMessageBox.information(self, "Not ready", "Load dataset and checkpoint first.")
            return

        sample_idx = self.sample_spin.value()

        try:
            self.status_label.setText(f"Analyzing sample {sample_idx}...")
            result = run_full_analysis(
                context=self.context,
                model=self.model,
                sample_idx=sample_idx,
                split=self.current_split,
            )

            fig = render_time_series_explanation(result)

            self.canvas.figure.clear()

            # Move axes/artists from returned fig into canvas by replacing figure reference
            self.canvas.figure = fig
            self.canvas.setParent(None)

            # Recreate canvas cleanly in layout
            parent_layout = self.layout()
            parent_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()

            self.canvas = FigureCanvas(fig)
            self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            parent_layout.addWidget(self.canvas, stretch=1)

            true_name = result.class_names[result.true_idx]
            pred_name = result.class_names[result.pred_idx]

            self.status_label.setText(
                f"Sample {sample_idx} | True={true_name} | Pred={pred_name} | Conf={result.confidence:.3f}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Analysis failed", str(e))
            self.status_label.setText("Analysis failed.")

    def _prev_sample(self):
        value = self.sample_spin.value()
        if value > 0:
            self.sample_spin.setValue(value - 1)
            self._analyze_current_sample()

    def _next_sample(self):
        value = self.sample_spin.value()
        if value < self.sample_spin.maximum():
            self.sample_spin.setValue(value + 1)
            self._analyze_current_sample()