"""Professional surveillance application with analytics."""

import sys
import os
import math
import time
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QEvent, QSize
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QTabWidget, QSpinBox,
                             QMessageBox, QDialog, QTableWidget, QTableWidgetItem,
                             QCheckBox, QGroupBox, QFormLayout, QComboBox,
                             QSlider, QScrollArea, QLineEdit, QSizePolicy,
                             QGridLayout)
from PyQt6.QtGui import QPixmap, QFont, QColor
from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime

from ui.video_thread_professional import ProfessionalVideoThread
from ui.zone_editor import ZoneEditorWidget
from ui.app_dark_advanced import SourceDialog, DARK_STYLESHEET
from vision.camera import Camera
from storage.zones_manager import ZonesManager
from vision.first_frame import get_first_frame


class VideoTile(QLabel):
    """Video surface that scales content to available space and notifies when selected."""

    clicked = pyqtSignal(int)

    def __init__(self, feed_id: int, parent: Optional[QWidget] = None,
                 preferred_width: int = 320, aspect_ratio: float = 16 / 9):
        super().__init__(parent)
        self.feed_id = feed_id
        self._last_pixmap: Optional[QPixmap] = None
        self._aspect_ratio = aspect_ratio
        self._preferred_width = max(160, int(preferred_width))
        self._preferred_height = max(90, int(self._preferred_width / self._aspect_ratio))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.resize_tile(self._preferred_width, self._preferred_height)
        self._apply_style(active=False)
        self.setText("Flux en attente...\nCliquez pour activer ce panneau")

    def _apply_style(self, active: bool):
        border = "#00ff00" if active else "#444444"
        self.setStyleSheet(
            f"border: 3px solid {border}; border-radius: 8px; background-color: #000000;"
        )

    def set_active(self, active: bool):
        self._apply_style(active)
        if active and self._last_pixmap is None:
            self.setText("Flux actif en attente...")

    def mousePressEvent(self, event):  # pragma: no cover - UI behavior
        self.clicked.emit(self.feed_id)
        super().mousePressEvent(event)

    def update_frame(self, image_bytes: bytes):
        pixmap = QPixmap()
        if not pixmap.loadFromData(image_bytes):
            return
        self._last_pixmap = pixmap
        self.setText("")
        self._apply_scaled_pixmap()

    def resizeEvent(self, event):  # pragma: no cover - UI behavior
        super().resizeEvent(event)
        self._apply_scaled_pixmap()

    def _apply_scaled_pixmap(self):
        if not self._last_pixmap:
            return
        size = self.size()
        if size.width() <= 0 or size.height() <= 0:
            return
        scaled = self._last_pixmap.scaled(
            size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled)

    def resize_tile(self, width: int, height: Optional[int] = None):
        width = max(100, int(width))
        if height is None:
            height = max(60, int(width / self._aspect_ratio))
        else:
            height = max(60, int(height))
        self._preferred_width = width
        self._preferred_height = height
        self.updateGeometry()
        self._apply_scaled_pixmap()

    def sizeHint(self):  # pragma: no cover - size negotiation is UI-specific
        return QSize(self._preferred_width, self._preferred_height)

    def minimumSizeHint(self):  # pragma: no cover - UI behavior
        return self.sizeHint()


@dataclass
class FeedContext:
    feed_id: int
    source: object
    source_type: int
    zone_profile_id: str
    video_label: VideoTile
    zones_config: Dict
    thread: Optional['ProfessionalVideoThread'] = None
    last_stats: Dict = field(default_factory=dict)
    last_frame_total: int = 0
    last_frame_index: int = 0
    last_source_fps: float = 0.0
    pending_speed: float = 1.0
    start_time: Optional[datetime] = None
    last_ui_frame_time: float = 0.0
    processing_mode: str = "background"

    @property
    def is_video_source(self) -> bool:
        return self.source_type == 2


class NoWheelSpinBox(QSpinBox):
    """Bloque la molette pour éviter des changements involontaires."""

    def wheelEvent(self, event):  # pragma: no cover - UI behavior
        event.ignore()


class NoWheelComboBox(QComboBox):
    """Bloque également la molette sur les listes déroulantes sensibles."""

    def wheelEvent(self, event):  # pragma: no cover - UI behavior
        event.ignore()


class AlertsWindow(QWidget):
    """Professional alerts monitoring window."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Système d'Alertes Professionnel")
        self.setGeometry(100, 100, 1000, 600)
        self.setStyleSheet(DARK_STYLESHEET)
        
        layout = QVBoxLayout()
        
        # Filter controls
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filtre:"))
        
        self.all_check = QCheckBox("Tous")
        self.all_check.setChecked(True)
        filter_layout.addWidget(self.all_check)
        self.all_check.stateChanged.connect(self._on_all_filter_changed)
        
        self.entry_check = QCheckBox("Entrées/Sorties")
        self.entry_check.setChecked(True)
        filter_layout.addWidget(self.entry_check)
        
        
        self.parking_check = QCheckBox("Stationnement")
        self.parking_check.setChecked(True)
        filter_layout.addWidget(self.parking_check)
        
        self.anomaly_check = QCheckBox("Anomalies")
        self.anomaly_check.setChecked(True)
        filter_layout.addWidget(self.anomaly_check)

        # Connect individual filters so 'Tous' reflects their state
        self.entry_check.stateChanged.connect(self._on_individual_filter_changed)
        self.parking_check.stateChanged.connect(self._on_individual_filter_changed)
        self.anomaly_check.stateChanged.connect(self._on_individual_filter_changed)
        
        filter_layout.addStretch()
        
        close_btn = QPushButton("Fermer")
        close_btn.clicked.connect(self.close)
        filter_layout.addWidget(close_btn)
        
        layout.addLayout(filter_layout)
        
        # Alerts table
        self.alerts_table = QTableWidget()
        self.alerts_table.setColumnCount(5)
        self.alerts_table.setHorizontalHeaderLabels([
            "Heure", "Type", "ID", "Caméra", "Message"
        ])
        self.alerts_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.alerts_table)
        
        self.setLayout(layout)
    
    def add_alert(self, alert_type: str, message: str, camera_id, track_id: int):
        """Add alert to table."""
        # Filtering: respect filter checkboxes
        typ = alert_type.upper()
        if typ == 'ENTRY' and not self.entry_check.isChecked():
            return
        if typ == 'PARKING' and not self.parking_check.isChecked():
            return
        if typ == 'ANOMALY' and not self.anomaly_check.isChecked():
            return

        row = self.alerts_table.rowCount()
        self.alerts_table.insertRow(row)

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.alerts_table.setItem(row, 0, QTableWidgetItem(timestamp))

        type_item = QTableWidgetItem(typ)
        if alert_type == "PARKING":
            type_item.setBackground(QColor(255, 165, 0))
        elif alert_type == "ANOMALY":
            type_item.setBackground(QColor(255, 0, 0))
        elif alert_type == "LOITERING":
            type_item.setBackground(QColor(255, 100, 0))
        else:
            type_item.setBackground(QColor(0, 150, 200))
        self.alerts_table.setItem(row, 1, type_item)
        
        self.alerts_table.setItem(row, 2, QTableWidgetItem(str(track_id)))
        self.alerts_table.setItem(row, 3, QTableWidgetItem(f"Cam {camera_id}"))
        self.alerts_table.setItem(row, 4, QTableWidgetItem(message))
        
        # Keep only last 100 rows
        while self.alerts_table.rowCount() > 100:
            self.alerts_table.removeRow(0)
        
        self.alerts_table.scrollToBottom()

    def _on_all_filter_changed(self, state):
        checked = state == 2
        self.entry_check.setChecked(checked)
        self.parking_check.setChecked(checked)
        self.anomaly_check.setChecked(checked)

    def _on_individual_filter_changed(self, _):
        # Keep 'Tous' in sync
        all_checked = self.entry_check.isChecked() and self.parking_check.isChecked() and self.anomaly_check.isChecked()
        self.all_check.setChecked(all_checked)
    
    def closeEvent(self, event):
        """Handle close without parent intervention."""
        event.accept()


class ProfessionalCameraTab(QWidget):
    """Professional camera monitoring tab."""
    
    def __init__(self, camera_id: int, source, source_type: int):
        super().__init__()
        self.camera_id = camera_id
        self.alerts_window = None
        self.feed_counter = 0
        self.feeds: Dict[int, FeedContext] = {}
        self.active_feed_id: Optional[int] = None
        self._slider_active = False
        self._slider_active_feed_id: Optional[int] = None
        self._pending_speed = 1.0
        self._hidden_classes = set()
        self._info_panel_defaults = {
            'fps': True,
            'objects': True,
            'entries': True,
            'exits': True,
            'parking': True,
            'forbidden': True
        }
        self._tile_min_width = 140
        self._tile_max_width = 640
        self._tile_aspect_ratio = 16 / 9
        self._max_feeds_per_tab: Optional[int] = None
        self._max_grid_columns = 4
        self._layout_presets = {
            1: {'rows': 1, 'columns': 1, 'slots': [(0, 0, 1, 1)]},
            2: {'rows': 1, 'columns': 2, 'slots': [(0, 0, 1, 1), (0, 1, 1, 1)]},
            3: {'rows': 2, 'columns': 2, 'slots': [(0, 0, 1, 1), (0, 1, 1, 1), (1, 0, 1, 2)]},
            4: {'rows': 2, 'columns': 2, 'slots': [(0, 0, 1, 1), (0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 1, 1)]},
            5: {'rows': 3, 'columns': 2, 'slots': [(0, 0, 1, 1), (0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 1, 1), (2, 0, 1, 2)]},
        }
        self._last_layout_rows = 0
        self._last_layout_cols = 0
        self._active_frame_interval = 1.0 / 24.0
        self._background_frame_interval = 1.0 / 8.0
        # Liste de modèles compatibles proposée à l'utilisateur
        self._available_models = [
            ("YOLOv8n (rapide)", "models/yolov8n.pt"),
            ("YOLOv8s", "models/yolov8s.pt"),
            ("YOLOv8m", "models/yolov8m.pt"),
            ("YOLOv8l", "models/yolov8l.pt"),
            ("YOLOv8x (précis)", "models/yolov8x.pt"),
        ]
        self._model_path = self._available_models[0][1]
        self._setup_ui()
        self._create_feed(source, source_type)
    
    def _setup_ui(self):
        """Setup professional UI."""
        layout = QHBoxLayout()
        
        # Left: Video grid (supports multiple simultaneous feeds)
        left_layout = QVBoxLayout()
        left_layout.setSpacing(12)

        self.video_grid_widget = QWidget()
        self.video_grid_layout = QGridLayout()
        self.video_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.video_grid_layout.setSpacing(12)
        self.video_grid_widget.setLayout(self.video_grid_layout)

        self.video_area = QScrollArea()
        self.video_area.setWidgetResizable(True)
        self.video_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.video_area.setWidget(self.video_grid_widget)
        self.video_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.video_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.video_area.viewport().installEventFilter(self)

        left_layout.addWidget(self.video_area)
        
        layout.addLayout(left_layout, 3)
        
        # Right: Control panel
        right_layout = QVBoxLayout()
        
        # Status
        status_group = QGroupBox("État du Système")
        status_layout = QFormLayout()
        
        self.status_label = QLabel("[STOP] Arretee")
        status_layout.addRow("Status:", self.status_label)
        
        self.uptime_label = QLabel("00:00:00")
        status_layout.addRow("Uptime:", self.uptime_label)
        
        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)
        
        # Statistics
        stats_group = QGroupBox("Statistiques")
        stats_layout = QFormLayout()
        
        self.fps_label = QLabel("0 FPS")
        stats_layout.addRow("FPS:", self.fps_label)
        
        self.detected_label = QLabel("0")
        stats_layout.addRow("Détectés:", self.detected_label)
        
        self.entries_label = QLabel("0")
        stats_layout.addRow("Entrees:", self.entries_label)
        
        self.exits_label = QLabel("0")
        stats_layout.addRow("Sorties:", self.exits_label)
        
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)
        
        # Alerts
        alerts_group = QGroupBox("Alertes")
        alerts_layout = QFormLayout()
        
        self.parking_label = QLabel("0")
        alerts_layout.addRow("Parking:", self.parking_label)
        
        self.unauthorized_label = QLabel("0")
        alerts_layout.addRow("Non-autorisées:", self.unauthorized_label)
        
        self.alerts_count_label = QLabel("0")
        alerts_layout.addRow("Total:", self.alerts_count_label)
        
        alerts_group.setLayout(alerts_layout)
        right_layout.addWidget(alerts_group)
        
        # Settings tabs
        settings_tabs = QTabWidget()

        # Standard settings tab
        standard_tab = QWidget()
        standard_layout = QVBoxLayout()
        standard_layout.setContentsMargins(0, 0, 0, 0)

        settings_group = QGroupBox("Configuration")
        settings_layout = QFormLayout()
        
        self.accuracy_spin = NoWheelSpinBox()
        self.accuracy_spin.setRange(10, 95)
        self.accuracy_spin.setValue(40)
        self.accuracy_spin.setSuffix("%")
        self.accuracy_spin.valueChanged.connect(self._on_accuracy_changed)
        settings_layout.addRow("Confiance:", self.accuracy_spin)
        
        self.show_accuracy_check = QCheckBox("Afficher confiance")
        self.show_accuracy_check.setChecked(True)
        self.show_accuracy_check.stateChanged.connect(self._on_show_accuracy_changed)
        settings_layout.addRow("", self.show_accuracy_check)

        self.show_boxes_check = QCheckBox("Afficher boîtes")
        self.show_boxes_check.setChecked(True)
        self.show_boxes_check.stateChanged.connect(self._on_show_boxes_changed)
        settings_layout.addRow("", self.show_boxes_check)

        self.show_labels_check = QCheckBox("Afficher étiquettes")
        self.show_labels_check.setChecked(True)
        self.show_labels_check.stateChanged.connect(self._on_show_labels_changed)
        settings_layout.addRow("", self.show_labels_check)

        self.show_zones_check = QCheckBox("Afficher zones")
        self.show_zones_check.setChecked(True)
        self.show_zones_check.stateChanged.connect(self._on_show_zones_changed)
        settings_layout.addRow("", self.show_zones_check)

        self.show_centers_check = QCheckBox("Afficher centre (counting)")
        self.show_centers_check.setChecked(False)
        self.show_centers_check.stateChanged.connect(self._on_show_centers_changed)
        settings_layout.addRow("", self.show_centers_check)
        
        settings_group.setLayout(settings_layout)
        standard_layout.addWidget(settings_group)
        standard_layout.addStretch()
        standard_tab.setLayout(standard_layout)

        # Advanced settings tab
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout()
        advanced_layout.setContentsMargins(0, 0, 0, 0)

        advanced_group = QGroupBox("Paramètres d'affichage avancés")
        advanced_group_layout = QVBoxLayout()

        self.show_info_panel_check = QCheckBox("Afficher panneau info (FPS, entrées, ...)")
        self.show_info_panel_check.setChecked(True)
        self.show_info_panel_check.stateChanged.connect(self._on_show_info_panel_changed)
        advanced_group_layout.addWidget(self.show_info_panel_check)

        advanced_group_layout.addWidget(QLabel("Éléments à afficher dans le panneau:"))
        self.info_field_checks = {}
        info_fields = [
            ("fps", "FPS"),
            ("objects", "Objets détectés"),
            ("entries", "Entrées"),
            ("exits", "Sorties"),
            ("parking", "Stationnement"),
            ("forbidden", "Zones interdites"),
        ]
        for key, label in info_fields:
            cb = QCheckBox(label)
            cb.setChecked(self._info_panel_defaults.get(key, True))
            cb.stateChanged.connect(self._on_info_fields_changed)
            advanced_group_layout.addWidget(cb)
            self.info_field_checks[key] = cb

        advanced_group_layout.addSpacing(10)
        advanced_group_layout.addWidget(QLabel("Masquer certaines classes (séparées par des virgules):"))
        self.hidden_classes_input = QLineEdit()
        self.hidden_classes_input.setPlaceholderText("ex: chair, bench, tv")
        advanced_group_layout.addWidget(self.hidden_classes_input)
        apply_hidden_btn = QPushButton("Appliquer filtres")
        apply_hidden_btn.clicked.connect(self._apply_hidden_classes)
        advanced_group_layout.addWidget(apply_hidden_btn)

        advanced_group.setLayout(advanced_group_layout)
        advanced_layout.addWidget(advanced_group)
        advanced_layout.addStretch()
        advanced_tab.setLayout(advanced_layout)

        model_tab = QWidget()
        model_layout = QVBoxLayout()
        model_layout.setContentsMargins(0, 0, 0, 0)

        model_group = QGroupBox("Modèle d'IA")
        model_group_layout = QFormLayout()

        self.model_combo = NoWheelComboBox()
        for label, path in self._available_models:
            self.model_combo.addItem(label, path)
        self.model_combo.addItem("Personnalisé", "__custom__")
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        model_group_layout.addRow("Profil:", self.model_combo)

        self.custom_model_input = QLineEdit()
        self.custom_model_input.setPlaceholderText("Chemin complet vers un poids .pt/.onnx")
        self.custom_model_input.setEnabled(False)
        model_group_layout.addRow("Fichier:", self.custom_model_input)

        self.model_status_label = QLabel("")
        self._refresh_model_status_label()
        model_group_layout.addRow("Actuel:", self.model_status_label)

        warning_label = QLabel("⚠️ Compatible uniquement avec les poids YOLOv8 (.pt Ultralytics)")
        warning_label.setWordWrap(True)
        warning_label.setStyleSheet("color: #ff8800; font-size: 12px;")
        model_group_layout.addRow("", warning_label)

        apply_model_btn = QPushButton("Appliquer modèle")
        apply_model_btn.clicked.connect(self._apply_model_selection)
        model_group_layout.addRow("", apply_model_btn)

        model_group.setLayout(model_group_layout)
        model_layout.addWidget(model_group)
        model_layout.addStretch()
        model_tab.setLayout(model_layout)

        settings_tabs.addTab(standard_tab, "Standard")
        settings_tabs.addTab(advanced_tab, "Avancé")
        settings_tabs.addTab(model_tab, "Modèle")
        right_layout.addWidget(settings_tabs)

        # Playback controls (videos only)
        playback_group = QGroupBox("Lecture vidéo")
        playback_layout = QFormLayout()

        self.speed_combo = NoWheelComboBox()
        for label, value in [("0.25x", 0.25), ("0.5x", 0.5), ("1x", 1.0), ("1.5x", 1.5), ("2x", 2.0), ("4x", 4.0)]:
            self.speed_combo.addItem(label, value)
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        playback_layout.addRow("Vitesse:", self.speed_combo)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setRange(0, 1000)
        self.timeline_slider.setPageStep(25)
        self.timeline_slider.sliderPressed.connect(self._on_seek_pressed)
        self.timeline_slider.sliderReleased.connect(self._on_seek_released)
        self.timeline_slider.valueChanged.connect(self._on_seek_value_changed)
        playback_layout.addRow("Position:", self.timeline_slider)

        self.playback_position_label = QLabel("Lecture en direct")
        playback_layout.addRow("", self.playback_position_label)

        playback_group.setLayout(playback_layout)
        playback_group.setEnabled(False)
        playback_group.setToolTip("Sélectionnez un flux vidéo pour activer ces contrôles")
        self.playback_group = playback_group
        right_layout.addWidget(playback_group)
        
        # Control buttons
        btn_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("▶ DÉMARRER")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self.start_camera)
        self.start_btn.setMinimumHeight(40)
        btn_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹ ARRÊTER")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        
        self.zones_btn = QPushButton("🎯 ZONES")
        self.zones_btn.setObjectName("zoneBtn")
        self.zones_btn.clicked.connect(self.open_zone_editor)
        self.zones_btn.setMinimumHeight(40)
        btn_layout.addWidget(self.zones_btn)
        
        self.preview_btn = QPushButton("👁 PREVIEW")
        self.preview_btn.clicked.connect(self.preview_first_frame)
        self.preview_btn.setMinimumHeight(40)
        btn_layout.addWidget(self.preview_btn)
        
        self.alerts_btn = QPushButton("🚨 ALERTES")
        self.alerts_btn.setMinimumHeight(40)
        self.alerts_btn.clicked.connect(self.show_alerts)
        btn_layout.addWidget(self.alerts_btn)

        self.add_feed_btn = QPushButton("➕ AJOUTER FLUX DANS L'ONGLET")
        self.add_feed_btn.setMinimumHeight(36)
        self.add_feed_btn.clicked.connect(self._add_additional_feed)
        btn_layout.addWidget(self.add_feed_btn)
        
        right_layout.addLayout(btn_layout)
        right_layout.addStretch()
        
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        right_scroll.setWidget(right_widget)

        layout.addWidget(right_scroll, 1)
        self.setLayout(layout)
        
        # Update timer for uptime
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_uptime)
        self._reset_playback_controls()

    def _create_feed(self, source, source_type):
        feed_id = self.feed_counter
        self.feed_counter += 1
        tile = VideoTile(feed_id, preferred_width=self._tile_max_width, aspect_ratio=self._tile_aspect_ratio)
        tile.clicked.connect(self._set_active_feed)
        context = FeedContext(
            feed_id=feed_id,
            source=source,
            source_type=source_type,
            zone_profile_id=str(source),
            video_label=tile,
            zones_config=ZonesManager.load_zones(str(source))
        )
        self.feeds[feed_id] = context
        self._reflow_video_tiles()
        self._sync_feed_limit_state()
        if self.active_feed_id is None:
            self._set_active_feed(feed_id)
        else:
            tile.set_active(False)
            self._update_status_badge()
            context.processing_mode = "background"

    def _reflow_video_tiles(self):
        count = self.video_grid_layout.count()
        for index in reversed(range(count)):
            item = self.video_grid_layout.takeAt(index)
            widget = item.widget()
            if widget:
                widget.setParent(None)

        feed_items = list(self.feeds.values())
        if not feed_items:
            placeholder = QLabel("Ajoutez un flux pour commencer")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: #888; border: 1px dashed #444; padding: 40px;")
            self.video_grid_layout.addWidget(placeholder, 0, 0)
            return

        layout_def = self._build_layout_blueprint(len(feed_items))
        base_width, base_height = self._calculate_tile_hints(layout_def['rows'], layout_def['columns'])

        for row in range(self._last_layout_rows):
            self.video_grid_layout.setRowStretch(row, 0)
        for col in range(self._last_layout_cols):
            self.video_grid_layout.setColumnStretch(col, 0)

        for row in range(layout_def['rows']):
            self.video_grid_layout.setRowStretch(row, 1)
        for col in range(layout_def['columns']):
            self.video_grid_layout.setColumnStretch(col, 1)

        for feed, slot in zip(feed_items, layout_def['slots']):
            row, col, row_span, col_span = slot
            pref_width = base_width * col_span
            pref_height = base_height * row_span
            feed.video_label.resize_tile(pref_width, pref_height)
            self.video_grid_layout.addWidget(feed.video_label, row, col, row_span, col_span)

        self._last_layout_rows = layout_def['rows']
        self._last_layout_cols = layout_def['columns']

    def _build_layout_blueprint(self, feed_count: int):
        preset = self._layout_presets.get(feed_count)
        if preset:
            return preset
        columns = max(1, min(self._max_grid_columns, int(math.ceil(math.sqrt(feed_count)))))
        rows = max(1, int(math.ceil(feed_count / columns)))
        slots = []
        idx = 0
        for row in range(rows):
            for col in range(columns):
                if idx >= feed_count:
                    break
                slots.append((row, col, 1, 1))
                idx += 1
        return {'rows': rows, 'columns': columns, 'slots': slots}

    def _calculate_tile_hints(self, rows: int, columns: int) -> tuple[int, int]:
        viewport = self.video_area.viewport()
        spacing_w = self.video_grid_layout.horizontalSpacing() or 0
        spacing_h = self.video_grid_layout.verticalSpacing() or spacing_w
        available_width = viewport.width() or self.video_area.width() or self._tile_max_width
        available_height = viewport.height() or self.video_area.height() or int(self._tile_max_width / self._tile_aspect_ratio)
        available_width = max(self._tile_min_width, available_width - spacing_w * max(0, columns - 1))
        available_height = max(80, available_height - spacing_h * max(0, rows - 1))

        column_width = max(120, available_width // max(1, columns))
        row_height = max(90, available_height // max(1, rows))

        base_width = min(self._tile_max_width, column_width)
        width_from_height = int(row_height * self._tile_aspect_ratio)
        if width_from_height > 0:
            base_width = min(base_width, width_from_height)
        base_width = max(self._tile_min_width, base_width)
        base_height = max(90, int(base_width / self._tile_aspect_ratio))
        return base_width, base_height

    def eventFilter(self, obj, event):  # pragma: no cover - UI behavior
        if obj == self.video_area.viewport() and event.type() == QEvent.Type.Resize:
            self._reflow_video_tiles()
        return super().eventFilter(obj, event)

    def _set_active_feed(self, feed_id: int):
        feed = self.feeds.get(feed_id)
        if not feed:
            return
        if self.active_feed_id == feed_id:
            return
        if self.active_feed_id is not None and self.active_feed_id in self.feeds:
            self.feeds[self.active_feed_id].video_label.set_active(False)
        self.active_feed_id = feed_id
        feed.video_label.set_active(True)
        self._sync_active_feed_ui(feed)
        self._apply_processing_modes()

    def _sync_active_feed_ui(self, feed: FeedContext):
        stats = feed.last_stats or {
            'fps': 0.0,
            'current_objects': 0,
            'entries': 0,
            'exits': 0,
            'parking_objects': 0,
            'parking_violations': 0,
            'forbidden_objects': 0,
            'unauthorized_zones': 0,
            'alerts': 0
        }
        self._refresh_stats_panel(stats)
        self._update_status_badge()
        self._update_uptime(force=True)
        self._sync_playback_controls_state(feed)
        self._refresh_timeline_labels(feed)
        self._update_controls_state()

    def _refresh_stats_panel(self, stats: Dict):
        self.fps_label.setText(f"{stats.get('fps', 0):.1f} FPS")
        self.detected_label.setText(str(stats.get('current_objects', 0)))
        self.entries_label.setText(str(stats.get('entries', 0)))
        self.exits_label.setText(str(stats.get('exits', 0)))
        parking_text = f"{stats.get('parking_objects', 0)} vehicules"
        if stats.get('parking_violations'):
            parking_text += f" | alertes: {stats['parking_violations']}"
        self.parking_label.setText(parking_text)
        forbidden_text = f"{stats.get('forbidden_objects', 0)} personnes"
        if stats.get('unauthorized_zones'):
            forbidden_text += f" | alertes: {stats['unauthorized_zones']}"
        self.unauthorized_label.setText(forbidden_text)
        self.alerts_count_label.setText(str(stats.get('alerts', 0)))

    def _sync_playback_controls_state(self, feed: Optional[FeedContext]):
        is_video = bool(feed and feed.is_video_source)
        self.playback_group.setEnabled(is_video)
        self.timeline_slider.setEnabled(is_video and bool(feed.thread))
        if not is_video:
            self.playback_position_label.setText("Lecture en direct")

    def _refresh_timeline_labels(self, feed: FeedContext):
        if feed.is_video_source and feed.last_frame_total > 0:
            self.playback_position_label.setText(
                self._format_position_text(feed.last_frame_index, feed.last_frame_total, feed.last_source_fps)
            )
        else:
            self.playback_position_label.setText("Lecture en direct")

    def _get_active_feed(self, required: bool = False) -> Optional[FeedContext]:
        feed = self.feeds.get(self.active_feed_id) if self.active_feed_id is not None else None
        if not feed and required:
            QMessageBox.warning(self, "Flux requis", "Aucun flux n'est sélectionné dans cet onglet.")
        return feed

    def _add_additional_feed(self):
        if self._max_feeds_per_tab is not None and len(self.feeds) >= self._max_feeds_per_tab:
            QMessageBox.information(
                self,
                "Limite atteinte",
                f"Vous pouvez afficher au maximum {self._max_feeds_per_tab} flux dans cet onglet."
            )
            return
        dialog = SourceDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.source is not None:
            self._create_feed(dialog.source, dialog.source_type)
            self._update_status_badge()
            self._apply_processing_modes()

    def _sync_feed_limit_state(self):
        if hasattr(self, 'add_feed_btn'):
            if self._max_feeds_per_tab is None:
                self.add_feed_btn.setEnabled(True)
                tooltip = ""
            else:
                can_add = len(self.feeds) < self._max_feeds_per_tab
                self.add_feed_btn.setEnabled(can_add)
                tooltip = "" if can_add else f"Limité à {self._max_feeds_per_tab} flux par onglet"
            self.add_feed_btn.setToolTip(tooltip)

    def _apply_processing_modes(self):
        for feed_id, feed in self.feeds.items():
            target_mode = "active" if feed_id == self.active_feed_id else "background"
            feed.processing_mode = target_mode
            if feed.thread:
                feed.thread.set_processing_priority(target_mode)

    def _running_feed_count(self) -> int:
        return sum(1 for feed in self.feeds.values() if feed.thread and feed.thread.isRunning())

    def _update_status_badge(self):
        total = len(self.feeds)
        running = self._running_feed_count()
        if running:
            self.status_label.setText(f"🟢 Flux actifs: {running}/{total}")
        else:
            self.status_label.setText(f"[STOP] {total} flux prêts")
            if self.update_timer.isActive():
                self.update_timer.stop()
                self._update_uptime(force=True)
        self.start_btn.setEnabled(running < total or total == 0)
        self.stop_btn.setEnabled(running > 0)

    def _update_controls_state(self):
        has_feed = self.active_feed_id is not None and self.active_feed_id in self.feeds
        for widget in (self.zones_btn, self.preview_btn, self.alerts_btn):
            widget.setEnabled(has_feed)
        if not has_feed:
            self._reset_playback_controls()
    
    def start_camera(self):
        """Start all stopped feeds within the tab."""
        if not self.feeds:
            QMessageBox.warning(self, "Flux manquant", "Ajoutez un flux avant de démarrer.")
            return

        started_any = False
        for feed in self.feeds.values():
            if feed.thread and feed.thread.isRunning():
                continue
            try:
                test_cap = Camera(feed.source)
                test_cap.release()
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Impossible d'ouvrir la source {feed.source}:\n{e}")
                continue

            thread = ProfessionalVideoThread(
                feed_id=feed.feed_id,
                camera_id=feed.source,
                zones_config=feed.zones_config,
                source_type=feed.source_type,
                model_path=self._model_path
            )
            thread.frame_ready.connect(self._on_frame_ready)
            thread.stats_ready.connect(self._on_stats_ready)
            thread.alert_triggered.connect(self._on_alert)
            thread.error_occurred.connect(self._on_thread_error)

            try:
                thread.start()
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Impossible de démarrer {feed.source}:\n{e}")
                continue

            feed.thread = thread
            feed.start_time = datetime.now()
            feed.pending_speed = self._pending_speed
            feed.thread.update_zones(feed.zones_config)
            feed.thread.set_processing_priority(feed.processing_mode)
            if feed.is_video_source:
                feed.thread.set_playback_speed(feed.pending_speed)
                if feed.feed_id == self.active_feed_id:
                    self.playback_position_label.setText("Chargement de la vidéo...")
            started_any = True

        if started_any:
            self._apply_overlay_settings()
            self._apply_advanced_settings()
            self._update_status_badge()
            self.update_timer.start(1000)
            self._sync_playback_controls_state(self._get_active_feed())
            self._apply_processing_modes()
        else:
            if any((not feed.thread) or (feed.thread and not feed.thread.isRunning()) for feed in self.feeds.values()):
                QMessageBox.information(self, "Flux", "Aucun flux n'a pu être démarré.")
    
    def stop_camera(self):
        """Stop camera monitoring."""
        stopped_any = False
        for feed in self.feeds.values():
            if feed.thread and feed.thread.isRunning():
                feed.thread.stop()
                feed.thread = None
                stopped_any = True
        if stopped_any:
            self.update_timer.stop()
            self.stop_btn.setEnabled(False)
            self._reset_playback_controls()
            self._update_uptime(force=True)
        self._update_status_badge()
    
    def preview_first_frame(self):
        """Preview first frame of source for zone editing."""
        feed = self._get_active_feed(required=True)
        if not feed:
            return
        frame = get_first_frame(feed.source)
        if frame is None:
            QMessageBox.warning(self, "Erreur", "Impossible de lire la première frame")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Prévisualisation - Éditeur de zones")
        dialog.setGeometry(100, 100, 1000, 700)
        dialog.setStyleSheet(DARK_STYLESHEET)
        
        layout = QVBoxLayout()
        
        zone_editor = ZoneEditorWidget(initial_frame=frame)
        zone_editor.zones = feed.zones_config.copy()
        zone_editor._update_display()
        
        def on_zones_done(zones):
            self._on_zones_modified(feed.feed_id, zones)
            dialog.close()
        
        zone_editor.zones_modified.connect(on_zones_done)
        layout.addWidget(zone_editor)
        dialog.setLayout(layout)
        dialog.exec()

    def open_zone_editor(self):
        """Open zone editor - now works during video."""
        feed = self._get_active_feed(required=True)
        if not feed:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Éditeur de zones")
        dialog.setGeometry(100, 100, 1000, 700)
        dialog.setStyleSheet(DARK_STYLESHEET)
        
        layout = QVBoxLayout()
        
        if not feed.thread or not feed.thread.isRunning():
            # Use preview button instead; this is now for live editing
            zone_editor = ZoneEditorWidget()
        else:
            dialog.setWindowTitle("Éditeur de zones (En direct)")
            zone_editor = ZoneEditorWidget(
                get_frame_callback=feed.thread.get_current_frame
            )
            zone_editor.zones = feed.zones_config.copy()
            zone_editor._update_display()
        
        def on_zones_done(zones):
            self._on_zones_modified(feed.feed_id, zones)
            dialog.close()
        
        zone_editor.zones_modified.connect(on_zones_done)
        layout.addWidget(zone_editor)
        dialog.setLayout(layout)
        dialog.exec()
    
    def show_alerts(self):
        """Show alerts window."""
        if self.alerts_window is None:
            self.alerts_window = AlertsWindow()
        self.alerts_window.show()
        self.alerts_window.raise_()
        self.alerts_window.activateWindow()
    
    def _update_uptime(self, force: bool = False):
        """Update uptime display for the active feed."""
        feed = self._get_active_feed()
        if feed and feed.start_time:
            elapsed = datetime.now() - feed.start_time
            hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.uptime_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        else:
            if force or not feed:
                self.uptime_label.setText("00:00:00")
    
    def _on_zones_modified(self, feed_id: int, zones: Dict):
        """Zone configuration updated for a specific feed."""
        feed = self.feeds.get(feed_id)
        if not feed:
            return
        feed.zones_config = zones.copy()
        ZonesManager.save_zones(zones, feed.zone_profile_id)
        if feed.thread:
            feed.thread.update_zones(zones)
    
    def _on_frame_ready(self, feed_id: int, image_bytes: bytes):
        """Update video display for the matching feed."""
        feed = self.feeds.get(feed_id)
        if not feed:
            return
        now = time.monotonic()
        interval = self._active_frame_interval if feed_id == self.active_feed_id else self._background_frame_interval
        if interval > 0 and (now - feed.last_ui_frame_time) < interval:
            return
        feed.last_ui_frame_time = now
        try:
            feed.video_label.update_frame(image_bytes)
        except Exception as e:
            print(f"[App] Error updating frame from bytes: {e}")
    
    def _on_stats_ready(self, feed_id: int, stats: Dict):
        """Update statistics display for the matching feed."""
        feed = self.feeds.get(feed_id)
        if not feed:
            return
        feed.last_stats = stats
        feed.last_frame_total = stats.get('frame_total') or 0
        feed.last_frame_index = stats.get('frame_index') or 0
        feed.last_source_fps = stats.get('source_fps') or 0.0
        if feed_id == self.active_feed_id:
            self._refresh_stats_panel(stats)
            self._update_playback_controls(feed)
    
    def _on_alert(self, feed_id: int, alert_type: str, message: str):
        """Handle alert trigger."""
        track_id = 0
        try:
            if "ID:" in message:
                track_id = int(message.split("ID:")[1].split()[0])
        except:
            pass
        
        if self.alerts_window and self.alerts_window.isVisible():
            camera_label = f"Tab {self.camera_id} | Flux {feed_id}"
            self.alerts_window.add_alert(alert_type, message, camera_label, track_id)
    
    def _on_thread_error(self, feed_id: int, error_msg: str):
        """Handle thread error for a specific feed."""
        print(f"[App] Thread error (feed {feed_id}): {error_msg}")
        QMessageBox.critical(self, "Erreur du fil d'exécution", f"Flux {feed_id}: {error_msg}")
        feed = self.feeds.get(feed_id)
        if feed and feed.thread:
            feed.thread = None
        self._update_status_badge()
    
    def _on_accuracy_changed(self, value: int):
        """Update accuracy."""
        for feed in self.feeds.values():
            if feed.thread:
                feed.thread.set_accuracy(value / 100.0)
    
    def _on_show_accuracy_changed(self, state):
        """Update show accuracy."""
        for feed in self.feeds.values():
            if feed.thread:
                feed.thread.set_show_accuracy(self.show_accuracy_check.isChecked())

    def _apply_overlay_settings(self):
        """Apply overlay toggles to the worker thread."""
        for feed in self.feeds.values():
            if not feed.thread:
                continue
            feed.thread.set_show_boxes(self.show_boxes_check.isChecked())
            feed.thread.set_show_labels(self.show_labels_check.isChecked())
            feed.thread.set_show_zones(self.show_zones_check.isChecked())
            feed.thread.set_show_centers(self.show_centers_check.isChecked())

    def _apply_advanced_settings(self):
        """Push advanced display settings to the worker thread."""
        info_states = self._get_info_field_states()
        self._info_panel_defaults.update(info_states)
        for feed in self.feeds.values():
            if not feed.thread:
                continue
            feed.thread.set_info_panel_visibility(self.show_info_panel_check.isChecked())
            feed.thread.set_info_panel_fields(info_states)
            feed.thread.set_hidden_classes(list(self._hidden_classes))

    def _get_info_field_states(self):
        states = {}
        if hasattr(self, 'info_field_checks'):
            for key, cb in self.info_field_checks.items():
                states[key] = cb.isChecked()
        else:
            states = self._info_panel_defaults.copy()
        return states

    def _on_show_info_panel_changed(self, _state):
        self._apply_advanced_settings()

    def _on_info_fields_changed(self, _state):
        self._apply_advanced_settings()

    def _apply_hidden_classes(self):
        self._hidden_classes = self._get_hidden_classes()
        for feed in self.feeds.values():
            if feed.thread:
                feed.thread.set_hidden_classes(list(self._hidden_classes))

    def _get_hidden_classes(self):
        text = self.hidden_classes_input.text() if hasattr(self, 'hidden_classes_input') else ""
        classes = [c.strip().lower() for c in text.split(',') if c.strip()]
        return set(classes)

    def _on_model_combo_changed(self, _index):
        if not hasattr(self, 'custom_model_input'):
            return
        is_custom = (self.model_combo.currentData() == "__custom__")
        self.custom_model_input.setEnabled(is_custom)
        if not is_custom:
            self.custom_model_input.clear()

    def _apply_model_selection(self):
        """Valide la sélection, sécurise le chemin puis relance le modèle côté worker."""
        if not hasattr(self, 'model_combo'):
            return
        selected = self.model_combo.currentData()
        if selected == "__custom__":
            path = self.custom_model_input.text().strip()
            if not path:
                QMessageBox.warning(self, "Modèle requis", "Veuillez renseigner le chemin du modèle personnalisé.")
                return
        else:
            path = selected or self._model_path
        if not path:
            return
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Modèle introuvable", f"Le fichier suivant est inaccessible:\n{path}")
            return
        if not path.lower().endswith(".pt"):
            QMessageBox.warning(self, "Modèle incompatible", "Seuls les poids YOLOv8 au format .pt sont pris en charge.")
            return
        # Mémoriser le chemin et prévenir chaque thread vidéo
        self._model_path = path
        self._refresh_model_status_label()
        for feed in self.feeds.values():
            if feed.thread:
                feed.thread.set_model_path(self._model_path)

    def _refresh_model_status_label(self):
        if hasattr(self, 'model_status_label'):
            self.model_status_label.setText(self._model_path)

    def _on_show_boxes_changed(self, _state):
        self._apply_overlay_settings()

    def _on_show_labels_changed(self, _state):
        self._apply_overlay_settings()

    def _on_show_zones_changed(self, _state):
        self._apply_overlay_settings()

    def _on_show_centers_changed(self, _state):
        self._apply_overlay_settings()

    def _apply_speed_setting(self):
        """Apply current playback speed to the thread."""
        if not hasattr(self, 'speed_combo'):
            return
        speed = self.speed_combo.currentData()
        if speed is None:
            speed = 1.0
        self._pending_speed = float(speed)
        feed = self._get_active_feed()
        if feed and feed.is_video_source:
            feed.pending_speed = self._pending_speed
            if feed.thread:
                feed.thread.set_playback_speed(feed.pending_speed)

    def _on_speed_changed(self, _index: int):
        """Handle speed combo change."""
        self._apply_speed_setting()

    def _on_seek_pressed(self):
        """Mark slider as user-controlled."""
        self._slider_active = True
        self._slider_active_feed_id = self.active_feed_id

    def _on_seek_value_changed(self, value: int):
        """Update preview label while dragging."""
        if not self._slider_active:
            return
        feed = self.feeds.get(self._slider_active_feed_id or -1)
        if not feed or not feed.is_video_source or feed.last_frame_total <= 0:
            return
        progress = value / max(1, self.timeline_slider.maximum())
        target_frame = int(progress * max(1, feed.last_frame_total - 1))
        self.playback_position_label.setText(
            self._format_position_text(target_frame, feed.last_frame_total, feed.last_source_fps)
        )

    def _on_seek_released(self):
        """Seek to selected position when slider released."""
        feed = self._get_active_feed()
        if not feed or not feed.is_video_source:
            self._slider_active = False
            return
        progress = self.timeline_slider.value() / max(1, self.timeline_slider.maximum())
        if feed.thread and feed.thread.isRunning():
            feed.thread.seek_to_progress(progress)
        self._slider_active = False
        self._slider_active_feed_id = None

    def _format_time(self, seconds: float) -> str:
        seconds = max(0, int(seconds))
        minutes, secs = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _format_position_text(self, frame_index: int, frame_total: int, fps: float) -> str:
        if fps and fps > 0 and frame_total > 0:
            current_seconds = frame_index / fps
            total_seconds = frame_total / fps
            return f"{self._format_time(current_seconds)} / {self._format_time(total_seconds)}"
        if frame_total > 0:
            return f"Frame {frame_index}/{frame_total}"
        return "Lecture en direct"

    def _update_playback_controls(self, feed: FeedContext):
        """Synchronize slider/labels with backend stats of the active feed."""
        if not feed.is_video_source:
            self.timeline_slider.setEnabled(False)
            self.playback_position_label.setText("Lecture en direct")
            return

        frame_total = feed.last_frame_total
        frame_index = feed.last_frame_index
        source_fps = feed.last_source_fps

        if frame_total <= 0:
            self.timeline_slider.setEnabled(False)
            self.playback_position_label.setText("Lecture en direct")
            return

        self.timeline_slider.setEnabled(True)
        if not self._slider_active:
            progress = frame_index / max(1, frame_total - 1)
            slider_value = int(progress * self.timeline_slider.maximum())
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(slider_value)
            self.timeline_slider.blockSignals(False)

        self.playback_position_label.setText(
            self._format_position_text(frame_index, frame_total, source_fps)
        )

    def _reset_playback_controls(self):
        """Reset slider/labels when playback stops."""
        if not hasattr(self, 'timeline_slider'):
            return
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(0)
        self.timeline_slider.blockSignals(False)
        self.timeline_slider.setEnabled(False)
        self.playback_position_label.setText("Lecture en direct")
        self._slider_active = False
        self._slider_active_feed_id = None
        # Keep last selected speed so it re-applies on next start
    
    def cleanup(self):
        """Cleanup."""
        self.update_timer.stop()
        for feed in self.feeds.values():
            if feed.thread and feed.thread.isRunning():
                feed.thread.stop()
                feed.thread = None
        self._reset_playback_controls()
        if self.alerts_window:
            self.alerts_window.close()


class ProfessionalApp(QWidget):
    """Professional surveillance application."""
    
    def __init__(self):
        super().__init__()
        self.camera_tabs: Dict[int, ProfessionalCameraTab] = {}
        self.tab_counter = 0
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup professional application UI."""
        self.setWindowTitle("[CAMERA] Systeme de Surveillance Professionnel - v2.0")
        self.setGeometry(0, 0, 1600, 1000)
        self.setStyleSheet(DARK_STYLESHEET)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Professional header
        header = QLabel("SYSTÈME DE SURVEILLANCE ET ANALYTICS PROFESSIONNEL v2.0")
        header_font = QFont()
        header_font.setPointSize(16)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setStyleSheet("color: #00ff00; padding: 10px;")
        layout.addWidget(header)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Control panel
        control_layout = QHBoxLayout()
        
        add_btn = QPushButton("[+] AJOUTER SOURCE")
        add_btn.clicked.connect(self._add_source)
        add_btn.setObjectName("startBtn")
        add_btn.setMinimumHeight(40)
        control_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("[X] SUPPRIMER")
        remove_btn.clicked.connect(self._remove_current_tab)
        remove_btn.setObjectName("stopBtn")
        remove_btn.setMinimumHeight(40)
        control_layout.addWidget(remove_btn)
        
        control_layout.addStretch()
        
        self.info_label = QLabel("Caméras actives: 0")
        self.info_label.setStyleSheet("color: #00ff00; font-weight: bold;")
        control_layout.addWidget(self.info_label)
        
        layout.addLayout(control_layout)
        
        self.setLayout(layout)
    
    def _add_source(self):
        """Add new source."""
        dialog = SourceDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.source is not None:
                try:
                    tab = ProfessionalCameraTab(self.tab_counter, dialog.source, dialog.source_type)
                    self.camera_tabs[self.tab_counter] = tab

                    type_names = {0: "CAM", 1: "STREAM", 2: "VIDEO", 3: "IMAGES"}
                    label = f"{type_names.get(dialog.source_type, 'SOURCE')} {self.tab_counter}"

                    self.tab_widget.addTab(tab, label)
                    self.tab_counter += 1
                    self._update_info()
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    QMessageBox.critical(self, "Erreur", f"Impossible d'ajouter la source:\n{e}")
    
    def _remove_current_tab(self):
        """Remove current tab."""
        index = self.tab_widget.currentIndex()
        if index >= 0:
            tab = self.tab_widget.widget(index)
            if isinstance(tab, ProfessionalCameraTab):
                tab.cleanup()
                self.camera_tabs.pop(tab.camera_id, None)
            self.tab_widget.removeTab(index)
            self._update_info()
    
    def _update_info(self):
        """Update info label."""
        active_feeds = sum(tab._running_feed_count() for tab in self.camera_tabs.values())
        total_tabs = len(self.camera_tabs)
        self.info_label.setText(f"Onglets: {total_tabs} | Flux actifs: {active_feeds}")


def main():
    """Launch application."""
    app = QApplication(sys.argv)
    window = ProfessionalApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()