from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QSpinBox, QMessageBox, QSizePolicy, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QGroupBox, QTextBrowser
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from app.services.data_service import load_split_context, get_dataset_size, build_window_rows
from app.services.model_service import build_and_load_model
from app.services.analysis_service import run_analysis
from app.plotting.time_series_renderer import render_analysis_result


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
        self.sample_count = 0

        self._build_ui()
        self._set_workspace_loaded(False)

        self.window_rows = []

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
        self.sample_spin.valueChanged.connect(self._on_sample_index_changed)
        self.sample_spin.editingFinished.connect(self._analyze_current_sample)
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

        # Info row
        info_row = QHBoxLayout()

        self.split_info = QLabel("Split: -")
        info_row.addWidget(self.split_info)

        self.samples_info = QLabel("Samples: -")
        info_row.addWidget(self.samples_info)

        self.current_sample_info = QLabel("Current sample: -")
        info_row.addWidget(self.current_sample_info)

        controls_row.addWidget(QLabel("View:"))
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems([
            "Combined explainer",
            "Sliding Window Occlusion",
        ])
        self.view_mode_combo.currentIndexChanged.connect(self._on_view_mode_changed)
        controls_row.addWidget(self.view_mode_combo)

        info_row.addStretch()
        root.addLayout(info_row)

        # Status
        self.status_label = QLabel("Select dataset and checkpoint, then click Load.")
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        root.addWidget(self.status_label)

        # Main content area: plot on the left, window table on the right
        # Use QSplitter so the user can resize both panes horizontally.
        self.content_splitter = QSplitter(Qt.Horizontal)
        self.content_splitter.setChildrenCollapsible(False)
        self.content_splitter.setHandleWidth(8)

        # Canvas (left)
        self.canvas = MplCanvas(self)
        self.content_splitter.addWidget(self.canvas)

        # Right side panel: navigation table on top, explainer description below
        self.side_panel = QWidget()
        self.side_panel_layout = QVBoxLayout(self.side_panel)
        self.side_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.side_panel_layout.setSpacing(10)

        self.window_table = QTableWidget()
        self.window_table.setColumnCount(5)
        self.window_table.setHorizontalHeaderLabels([
            "Sample idx", "Vehicle", "Start time", "End time", "True"
        ])
        self.window_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.window_table.setSelectionMode(QTableWidget.SingleSelection)
        self.window_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.window_table.verticalHeader().setVisible(False)
        self.window_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.window_table.horizontalHeader().setStretchLastSection(True)
        self.window_table.cellDoubleClicked.connect(self._on_window_table_double_clicked)
        self.side_panel_layout.addWidget(self.window_table, stretch=3)

        self.methods_group = QGroupBox("O použitých metódach")
        methods_layout = QVBoxLayout(self.methods_group)

        self.methods_description = QTextBrowser()
        self.methods_description.setReadOnly(True)
        self.methods_description.setOpenExternalLinks(False)
        self.methods_description.setHtml(
            """
            <p>
                Tieto vizualizácie ukazujú, <b>ktoré časti vstupu boli pre model dôležité</b>
                pri jeho rozhodovaní. Výraznejšie zvýraznenie znamená väčší vplyv na výsledok.
            </p>
            <p><b>Feature Occlusion</b><br>
                Sleduje, ako sa zmení výsledok, keď sa skryje celý vstupný signál.
                Ak sa predikcia výrazne zhorší, daný signál bol dôležitý.
            </p>
            <p><b>Integrated Gradients</b><br>
                Ukazuje, ktoré konkrétne časti daného signálu najviac prispeli k výsledku modelu.
            </p>
            <p><b>Grad-CAM</b><br>
                Zvýrazňuje časové úseky naprieč signálmi, na ktoré sa model pri rozhodovaní najviac sústredil.
            </p>
            <p><b>Sliding Window Occlusion</b><br>
                Postupne zakrýva krátke úseky signálu a sleduje, ktoré z nich najviac ovplyvnia výsledok.
            </p>
            <p><b>Mean Window Occlusion</b><br>
                Predstavuje priemerný výsledok Sliding Window Occlusion pre daný signál.
            </p>
            <p><b>Window Overlap</b><br>
                Postupne zakrýva rovnaký časový úsek vo všetkých signáloch naraz a sleduje, ako to ovplyvní výsledok modelu.
            </p>
            <p><b>Poznámka:</b><br>
                Tieto metódy nevysvetľujú, čo je objektívne správne, ale to,
                <b>čo bolo dôležité pre samotný model</b>.
            </p>
            """
        )
        methods_layout.addWidget(self.methods_description)

        self.side_panel_layout.addWidget(self.methods_group, stretch=2)

        # optional: fixed/min width so it behaves like a side panel
        self.side_panel.setMinimumWidth(430)
        self.side_panel.setMaximumWidth(650)

        self.content_splitter.addWidget(self.side_panel)
        self.content_splitter.setStretchFactor(0, 4)
        self.content_splitter.setStretchFactor(1, 2)
        self.content_splitter.setSizes([950, 450])

        root.addWidget(self.content_splitter, stretch=1)

    def _set_workspace_loaded(self, loaded: bool):
        if not loaded:
            self.sample_spin.setEnabled(False)
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.analyze_btn.setEnabled(False)
        else:
            self._update_navigation_buttons()

    def _set_busy(self, busy: bool):
        if busy:
            QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QGuiApplication.restoreOverrideCursor()

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

    def _validate_paths(self):
        dataset_path = self.dataset_edit.text().strip()
        checkpoint_path = self.checkpoint_edit.text().strip()

        if not dataset_path:
            raise ValueError("Please select a dataset path.")

        if not checkpoint_path:
            raise ValueError("Please select a checkpoint path.")

        # Dataset can be either a file or a stem, so validate softly
        dataset_obj = Path(dataset_path)
        stem_exists = Path(dataset_path + "_vehicles.pcl").exists()
        file_exists = dataset_obj.exists()

        if not file_exists and not stem_exists:
            raise FileNotFoundError(
                "Dataset path not found. Select either the dataset stem or a real *_vehicles.pcl file."
            )

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError("Checkpoint file not found.")

        return dataset_path, checkpoint_path

    def _load_workspace(self):
        try:
            dataset_path, checkpoint_path = self._validate_paths()

            self._set_busy(True)
            self.status_label.setText("Loading dataset...")
            self.repaint()

            self.context = load_split_context(dataset_path)

            self.status_label.setText("Loading checkpoint...")
            self.repaint()

            self.model = build_and_load_model(checkpoint_path)

            self.sample_count = get_dataset_size(self.context, split=self.current_split)
            if self.sample_count <= 0:
                raise ValueError(f"No samples available in split '{self.current_split}'.")

            self.sample_spin.setMaximum(self.sample_count - 1)
            self.sample_spin.setValue(0)

            self.split_info.setText(f"Split: {self.current_split}")
            self.samples_info.setText(f"Samples: {self.sample_count}")
            self.current_sample_info.setText("Current sample: 0")

            self.status_label.setText(
                f"Workspace loaded. Split={self.current_split}, samples={self.sample_count}. "
                "Press Analyze to render the current sample."
            )

            self._set_workspace_loaded(True)
            self._update_navigation_buttons()

            self._populate_window_table()

            self._analyze_current_sample()

        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))
            self.status_label.setText("Load failed.")
            self._set_workspace_loaded(False)
        finally:
            self._set_busy(False)

    def _show_figure(self, fig):
        current_sizes = self.content_splitter.sizes()

        self.canvas.setParent(None)
        self.canvas.deleteLater()

        self.canvas = FigureCanvas(fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.content_splitter.insertWidget(0, self.canvas)

        if current_sizes and len(current_sizes) == 2:
            self.content_splitter.setSizes(current_sizes)

    def _analyze_current_sample(self):
        if self.context is None or self.model is None:
            QMessageBox.information(self, "Not ready", "Load dataset and checkpoint first.")
            return

        sample_idx = self.sample_spin.value()

        try:
            self._set_busy(True)
            self.status_label.setText(f"Analyzing sample {sample_idx}...")
            self.repaint()
            mode = self._current_view_mode()

            result = run_analysis(
                context=self.context,
                model=self.model,
                sample_idx=sample_idx,
                split=self.current_split,
                mode=mode,
            )

            fig = render_analysis_result(result)
            self._show_figure(fig)

            true_name = result.class_names[result.true_idx]
            pred_name = result.class_names[result.pred_idx]

            self.current_sample_info.setText(f"Current sample: {sample_idx}")
            self.status_label.setText(
                f"Sample {sample_idx} | True={true_name} | Pred={pred_name} | Conf={result.confidence:.3f}"
            )
            self._update_navigation_buttons()

            if 0 <= sample_idx < self.window_table.rowCount():
                self.window_table.selectRow(sample_idx)

        except Exception as e:
            QMessageBox.critical(self, "Analysis failed", str(e))
            self.status_label.setText("Analysis failed.")
        finally:
            self._set_busy(False)
            self._update_navigation_buttons()

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

    def _on_sample_index_changed(self, value):
        self.current_sample_info.setText(f"Current sample: {value}")
        self._update_navigation_buttons()

    def _update_navigation_buttons(self):
        loaded = self.context is not None and self.model is not None and self.sample_count > 0
        current = self.sample_spin.value()

        self.sample_spin.setEnabled(loaded)
        self.analyze_btn.setEnabled(loaded)

        self.prev_btn.setEnabled(loaded and current > 0)
        self.next_btn.setEnabled(loaded and current < self.sample_spin.maximum())

    def _current_view_mode(self):
        text = self.view_mode_combo.currentText()
        if text == "Combined explainer":
            return "combined"
        if text == "Sliding Window Occlusion":
            return "sliding_window"
        return "combined"

    def _on_view_mode_changed(self, _index):
        if self.context is None or self.model is None:
            return
        self._analyze_current_sample()

    def _on_window_table_double_clicked(self, row, _column):
        if row < 0 or row >= len(self.window_rows):
            return

        sample_idx = self.window_rows[row].sample_idx
        self.sample_spin.setValue(sample_idx)
        self._analyze_current_sample()

    def _populate_window_table(self):
        if self.context is None:
            self.window_table.setRowCount(0)
            self.window_rows = []
            return

        self.window_rows = build_window_rows(self.context, split=self.current_split)
        self.window_table.setRowCount(len(self.window_rows))

        for row_idx, row in enumerate(self.window_rows):
            self.window_table.setItem(row_idx, 0, QTableWidgetItem(str(row.sample_idx)))
            self.window_table.setItem(row_idx, 1, QTableWidgetItem(row.vehicle))
            self.window_table.setItem(row_idx, 2, QTableWidgetItem(row.start_time))
            self.window_table.setItem(row_idx, 3, QTableWidgetItem(row.end_time))
            self.window_table.setItem(row_idx, 4, QTableWidgetItem(row.true_label))

        if self.window_rows:
            self.window_table.selectRow(0)