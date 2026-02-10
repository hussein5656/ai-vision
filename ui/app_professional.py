"""Professional surveillance application with analytics."""

import sys
import os
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QTabWidget, QSpinBox,
                             QMessageBox, QDialog, QTableWidget, QTableWidgetItem,
                             QCheckBox, QGroupBox, QFormLayout, QComboBox,
                             QSlider, QScrollArea, QLineEdit)
from PyQt6.QtGui import QPixmap, QFont, QColor
from typing import Dict, Optional
from datetime import datetime

from ui.video_thread_professional import ProfessionalVideoThread
from ui.zone_editor import ZoneEditorWidget
from ui.app_dark_advanced import SourceDialog, DARK_STYLESHEET
from vision.camera import Camera
from storage.zones_manager import ZonesManager
from vision.first_frame import get_first_frame


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
        self.video_label.setText("En attente du flux vidéo...")  # Message d'attente pour le flux vidéo
        
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
    
    def add_alert(self, alert_type: str, message: str, camera_id: int, track_id: int):
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
        self.source = source
        self.source_type = source_type
        self.is_video_source = (self.source_type == 2)
        self.thread: Optional['ProfessionalVideoThread'] = None
        self.zone_profile_id = str(self.source)
        self.zones_config = ZonesManager.load_zones(self.zone_profile_id)
        self.alerts_window = None
        self._slider_active = False
        self._last_frame_total = 0
        self._last_source_fps = 0
        self._last_frame_index = 0
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
        self.start_time = None
    
    def _setup_ui(self):
        """Setup professional UI."""
        layout = QHBoxLayout()
        
        # Left: Video display
        left_layout = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(650, 490)
        self.video_label.setStyleSheet(
            "border: 3px solid #00ff00; background-color: #000000; border-radius: 4px;"
        )
        self.video_label.setText("En attente du flux vidéo...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.video_label)
        
        layout.addLayout(left_layout, 2)
        
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
        playback_group.setEnabled(self.is_video_source)
        if not self.is_video_source:
            playback_group.setToolTip("Contrôles disponibles uniquement pour les vidéos enregistrées")
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
        if self.is_video_source:
            self._reset_playback_controls()
    
    def start_camera(self):
        """Start professional camera monitoring."""
        if self.thread and self.thread.isRunning():
            QMessageBox.warning(self, "En cours", "Cette caméra est déjà active")
            return
        
        try:
            # Quick pre-check: attempt to open the source to fail fast for invalid videos
            try:
                test_cap = Camera(self.source)
                test_cap.release()
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Impossible d'ouvrir la source:\n{e}")
                self.status_label.setText("🔴 Erreur")
                return

            self.status_label.setText("🟡 Démarrage...")
            self.start_time = datetime.now()

            # On démarre le worker vidéo avec les zones et le modèle sélectionné
            self.thread = ProfessionalVideoThread(
                camera_id=self.source,
                zones_config=self.zones_config,
                source_type=self.source_type,
                model_path=self._model_path
            )
            
            self.thread.frame_ready.connect(self._on_frame_ready)
            self.thread.stats_ready.connect(self._on_stats_ready)
            self.thread.alert_triggered.connect(self._on_alert)
            self.thread.error_occurred.connect(self._on_thread_error)
            
            try:
                self.thread.start()
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Impossible de démarrer la source:\n{e}")
                self.status_label.setText("🔴 Erreur")
                return
            
            # Activate zones rules after thread starts
            self.thread.update_zones(self.zones_config)
            self._apply_speed_setting()
            self._apply_overlay_settings()
            self._apply_advanced_settings()
            
            self.status_label.setText("🟢 Active")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.zones_btn.setEnabled(True)
            self.update_timer.start(1000)
            if self.is_video_source:
                self.timeline_slider.setEnabled(True)
                self.playback_position_label.setText("Chargement de la vidéo...")
            
        except Exception as e:
            error_msg = f"Impossible de démarrer:\n{e}"
            print(f"[App] {error_msg}")
            QMessageBox.critical(self, "Erreur", error_msg)
            self.status_label.setText("🔴 Erreur")
    
    def stop_camera(self):
        """Stop camera monitoring."""
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.update_timer.stop()
            self.status_label.setText("🔴 Arrêtée")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self._reset_playback_controls()
    
    def preview_first_frame(self):
        """Preview first frame of source for zone editing."""
        frame = get_first_frame(self.source)
        if frame is None:
            QMessageBox.warning(self, "Erreur", "Impossible de lire la première frame")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Prévisualisation - Éditeur de zones")
        dialog.setGeometry(100, 100, 1000, 700)
        dialog.setStyleSheet(DARK_STYLESHEET)
        
        layout = QVBoxLayout()
        
        zone_editor = ZoneEditorWidget(initial_frame=frame)
        zone_editor.zones = self.zones_config.copy()
        zone_editor._update_display()
        
        def on_zones_done(zones):
            self._on_zones_modified(zones)
            dialog.close()
        
        zone_editor.zones_modified.connect(on_zones_done)
        layout.addWidget(zone_editor)
        dialog.setLayout(layout)
        dialog.exec()

    def open_zone_editor(self):
        """Open zone editor - now works during video."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Éditeur de zones")
        dialog.setGeometry(100, 100, 1000, 700)
        dialog.setStyleSheet(DARK_STYLESHEET)
        
        layout = QVBoxLayout()
        
        if not self.thread or not self.thread.isRunning():
            # Use preview button instead; this is now for live editing
            zone_editor = ZoneEditorWidget()
        else:
            dialog.setWindowTitle("Éditeur de zones (En direct)")
            zone_editor = ZoneEditorWidget(
                get_frame_callback=self.thread.get_current_frame
            )
            zone_editor.zones = self.zones_config.copy()
            zone_editor._update_display()
        
        def on_zones_done(zones):
            self._on_zones_modified(zones)
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
    
    def _update_uptime(self):
        """Update uptime display."""
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.uptime_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
    
    def _on_zones_modified(self, zones: Dict):
        """Zone configuration updated."""
        self.zones_config = zones.copy()
        ZonesManager.save_zones(zones, self.zone_profile_id)
        if self.thread:
            self.thread.update_zones(zones)
    
    def _on_frame_ready(self, image_bytes: bytes):
        """Update video display from JPEG bytes."""
        try:
            pixmap = QPixmap()
            pixmap.loadFromData(image_bytes)
            scaled = pixmap.scaledToWidth(650, Qt.TransformationMode.SmoothTransformation)
            self.video_label.setPixmap(scaled)
        except Exception as e:
            print(f"[App] Error updating frame from bytes: {e}")
    
    def _on_stats_ready(self, stats: Dict):
        """Update statistics display."""
        self.fps_label.setText(f"{stats['fps']:.1f} FPS")
        self.detected_label.setText(str(stats['current_objects']))
        self.entries_label.setText(str(stats['entries']))
        self.exits_label.setText(str(stats['exits']))
        parking_text = f"{stats['parking_objects']} vehicules"
        if stats['parking_violations']:
            parking_text += f" | alertes: {stats['parking_violations']}"
        self.parking_label.setText(parking_text)

        forbidden_text = f"{stats['forbidden_objects']} personnes"
        if stats['unauthorized_zones']:
            forbidden_text += f" | alertes: {stats['unauthorized_zones']}"
        self.unauthorized_label.setText(forbidden_text)
        self.alerts_count_label.setText(str(stats['alerts']))
        self._update_playback_controls(stats)
    
    def _on_alert(self, alert_type: str, message: str):
        """Handle alert trigger."""
        track_id = 0
        try:
            if "ID:" in message:
                track_id = int(message.split("ID:")[1].split()[0])
        except:
            pass
        
        if self.alerts_window and self.alerts_window.isVisible():
            self.alerts_window.add_alert(alert_type, message, self.camera_id, track_id)
    
    def _on_thread_error(self, error_msg: str):
        """Handle thread error."""
        print(f"[App] Thread error: {error_msg}")
        QMessageBox.critical(self, "Erreur du fil d'exécution", error_msg)
        self.status_label.setText("🔴 Erreur")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def _on_accuracy_changed(self, value: int):
        """Update accuracy."""
        if self.thread:
            self.thread.set_accuracy(value / 100.0)
    
    def _on_show_accuracy_changed(self, state):
        """Update show accuracy."""
        if self.thread:
            self.thread.set_show_accuracy(self.show_accuracy_check.isChecked())

    def _apply_overlay_settings(self):
        """Apply overlay toggles to the worker thread."""
        if not self.thread:
            return
        self.thread.set_show_boxes(self.show_boxes_check.isChecked())
        self.thread.set_show_labels(self.show_labels_check.isChecked())
        self.thread.set_show_zones(self.show_zones_check.isChecked())
        self.thread.set_show_centers(self.show_centers_check.isChecked())

    def _apply_advanced_settings(self):
        """Push advanced display settings to the worker thread."""
        info_states = self._get_info_field_states()
        self._info_panel_defaults.update(info_states)
        if self.thread:
            self.thread.set_info_panel_visibility(self.show_info_panel_check.isChecked())
            self.thread.set_info_panel_fields(info_states)
            self.thread.set_hidden_classes(list(self._hidden_classes))

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
        if self.thread:
            self.thread.set_hidden_classes(list(self._hidden_classes))

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
        if not path.lower().endswith(".pt"):
            QMessageBox.warning(self, "Modèle incompatible", "Seuls les poids YOLOv8 au format .pt sont pris en charge.")
            return
        # Mémoriser le chemin et prévenir le thread vidéo pour qu'il recharge YOLO
        self._model_path = path
        self._refresh_model_status_label()
        if self.thread:
            self.thread.set_model_path(self._model_path)

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
        if self.is_video_source and self.thread:
            self.thread.set_playback_speed(self._pending_speed)

    def _on_speed_changed(self, _index: int):
        """Handle speed combo change."""
        self._apply_speed_setting()

    def _on_seek_pressed(self):
        """Mark slider as user-controlled."""
        self._slider_active = True

    def _on_seek_value_changed(self, value: int):
        """Update preview label while dragging."""
        if not (self.is_video_source and self._slider_active and self._last_frame_total > 0):
            return
        progress = value / max(1, self.timeline_slider.maximum())
        target_frame = int(progress * max(1, self._last_frame_total - 1))
        self.playback_position_label.setText(
            self._format_position_text(target_frame, self._last_frame_total, self._last_source_fps)
        )

    def _on_seek_released(self):
        """Seek to selected position when slider released."""
        if not self.is_video_source:
            self._slider_active = False
            return
        progress = self.timeline_slider.value() / max(1, self.timeline_slider.maximum())
        if self.thread and self.thread.isRunning():
            self.thread.seek_to_progress(progress)
        self._slider_active = False

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

    def _update_playback_controls(self, stats: Dict):
        """Synchronize slider/labels with backend stats."""
        if not self.is_video_source:
            return
        frame_total = stats.get('frame_total') or 0
        frame_index = stats.get('frame_index') or 0
        source_fps = stats.get('source_fps') or 0
        self._last_frame_total = frame_total
        self._last_frame_index = frame_index
        self._last_source_fps = source_fps

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
        if not self.is_video_source or not hasattr(self, 'timeline_slider'):
            return
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(0)
        self.timeline_slider.blockSignals(False)
        self.timeline_slider.setEnabled(False)
        self.playback_position_label.setText("Lecture en direct")
        self._last_frame_total = 0
        self._last_frame_index = 0
        self._slider_active = False
        # Keep last selected speed so it re-applies on next start
    
    def cleanup(self):
        """Cleanup."""
        self.update_timer.stop()
        if self.thread and self.thread.isRunning():
            self.thread.stop()
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
            self.tab_widget.removeTab(index)
            self._update_info()
    
    def _update_info(self):
        """Update info label."""
        active = sum(1 for tab in self.camera_tabs.values() 
                     if tab.thread and tab.thread.isRunning())
        total = len(self.camera_tabs)
        self.info_label.setText(f"Caméras: {total} | Actives: {active}")


def main():
    """Launch application."""
    app = QApplication(sys.argv)
    window = ProfessionalApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()