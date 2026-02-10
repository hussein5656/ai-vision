"""Simplified and intuitive zone editor for easy zone drawing."""

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QComboBox, QMessageBox)
from PyQt6.QtGui import QPixmap, QImage, QFont, QPainter, QPen, QColor
from typing import Dict, List, Callable, Optional


class ZoneEditorWidget(QWidget):
    """Ultra-simple zone editor with intuitive UI."""
    
    zones_modified = pyqtSignal(dict)
    
    def __init__(self, get_frame_callback: Optional[Callable] = None, initial_frame=None):
        super().__init__()
        self.get_frame_callback = get_frame_callback
        self.initial_frame = initial_frame
        
        # Zone types mapping
        self.zone_types = {
            'Ligne de comptage': 'counting_lines',
            'Zone de stationnement': 'parking_zones',
            'Zone interdite': 'forbidden_zones',
            'Zone de flânerie': 'loitering_zones',
        }
        
        self.zones = {
            'counting_lines': [],
            'forbidden_zones': [],
            'parking_zones': [],
            'loitering_zones': [],
            'restricted_areas': []
        }
        
        self.current_zone_type = 'counting_lines'
        self.current_points = []
        self.frame = initial_frame
        self.drawing_mode = False
        
        self._setup_ui()
        self._update_display()
        
        # Timer for live frame updates
        if self.get_frame_callback:
            self.timer = QTimer()
            self.timer.timeout.connect(self._refresh_frame)
            self.timer.start(100)
    
    def _setup_ui(self):
        """Setup simplified UI."""
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel(
            "[*] Cliquez sur l'image pour ajouter des points\n"
            "[LIGNE] Pour une ligne de comptage, placez DEUX points (début/fin)\n"
            "[2x] Double-cliquez ou appuyez 'Terminer' pour finaliser la zone\n"
            "[EDIT] Vous pouvez dessiner plusieurs zones avant d'appliquer"
        )
        instructions.setStyleSheet("color: #cccccc; padding: 10px; background: #333333; border-radius: 5px;")
        instructions.setFont(QFont("Arial", 9))
        layout.addWidget(instructions)
        
        # Zone type selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Type:"))
        
        self.zone_combo = QComboBox()
        self.zone_combo.addItems(self.zone_types.keys())
        self.zone_combo.currentTextChanged.connect(self._on_type_changed)
        selector_layout.addWidget(self.zone_combo)

        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["Horizontal", "Vertical"])
        self.direction_combo.currentTextChanged.connect(self._on_direction_changed)
        selector_layout.addWidget(self.direction_combo)
        self.direction_combo.setVisible(False)
        
        # Quick info
        self.info_label = QLabel("En attente...")
        self.info_label.setStyleSheet("color: #ffaa00; font-weight: bold;")
        selector_layout.addStretch()
        selector_layout.addWidget(self.info_label)
        
        layout.addLayout(selector_layout)
        
        # Canvas - using ZoneCanvas for interactive drawing
        self.canvas = ZoneCanvas()
        self.canvas.clicked.connect(self._on_canvas_click)
        self.canvas.double_clicked.connect(self._on_canvas_double_click)
        layout.addWidget(self.canvas)
        
        # Simple button controls
        button_layout = QHBoxLayout()
        
        btn_undo = QPushButton("[UNDO] Annuler dernier point")
        btn_undo.clicked.connect(self._undo_point)
        button_layout.addWidget(btn_undo)
        
        btn_finish = QPushButton("[DONE] Terminer cette zone")
        btn_finish.setStyleSheet("background: #0077cc; color: white; font-weight: bold;")
        btn_finish.clicked.connect(self._finish_zone)
        button_layout.addWidget(btn_finish)
        
        btn_clear = QPushButton("[DEL] Tout effacer")  # Remet tout à zéro
        btn_clear.clicked.connect(self._clear_all)
        button_layout.addWidget(btn_clear)
        
        btn_apply = QPushButton("[OK] APPLIQUER")
        btn_apply.setStyleSheet("background: #00aa00; color: white; font-weight: bold; padding: 10px;")
        btn_apply.clicked.connect(self._apply_zones)
        button_layout.addWidget(btn_apply)
        
        layout.addLayout(button_layout)
        
        # Status bar
        self.status_label = QLabel("Prêt")
        self.status_label.setStyleSheet("color: #00cc00; padding: 5px;")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def _on_type_changed(self):
        """Zone type changed."""
        self.current_zone_type = self.zone_types[self.zone_combo.currentText()]
        self.current_points = []
        self.direction_combo.setVisible(self.current_zone_type == 'counting_lines')
        self._update_display()
        zone_name = self.zone_combo.currentText()
        self.status_label.setText(f"[EDIT] Pret a dessiner: {zone_name}")

    def _on_direction_changed(self):
        self.current_points = []
        self._update_display()
    
    def _on_canvas_click(self, x: int, y: int):
        """Canvas clicked."""
        if self.frame is not None:
            # Map screen coordinates to frame coordinates
            frame_x, frame_y = self._screen_to_frame(x, y)
            self.current_points.append([frame_x, frame_y])
            self.info_label.setText(
                f"Points: {len(self.current_points)} {'(minimum 3)' if len(self.current_points) < 3 else '[OK]'}"
            )
            self._update_display()
    
    def _on_canvas_double_click(self, x: int, y: int):
        """Canvas double-clicked = finish zone."""
        self._finish_zone()
    
    def _screen_to_frame(self, screen_x, screen_y):
        """Convert screen coordinates to frame coordinates."""
        if self.canvas.pixmap is None or self.frame is None:
            return int(screen_x), int(screen_y)

        frame_h, frame_w = self.frame.shape[:2]

        scale = getattr(self.canvas, "_scale", None)
        offset_x, offset_y = getattr(self.canvas, "_offset", (0, 0))

        if not scale or scale <= 0:
            canvas_w = self.canvas.width()
            canvas_h = self.canvas.height()
            scale = min(canvas_w / frame_w, canvas_h / frame_h)
            offset_x = (canvas_w - int(frame_w * scale)) // 2
            offset_y = (canvas_h - int(frame_h * scale)) // 2

        frame_x = int((screen_x - offset_x) / scale)
        frame_y = int((screen_y - offset_y) / scale)

        frame_x = max(0, min(frame_x, frame_w - 1))
        frame_y = max(0, min(frame_y, frame_h - 1))

        return frame_x, frame_y
    
    def _undo_point(self):
        """Remove last point."""
        if self.current_points:
            self.current_points.pop()
            self.info_label.setText(f"Points: {len(self.current_points)}")
            self._update_display()
            self.status_label.setText("Point annulé")
    
    def _finish_zone(self):
        """Finish current zone."""
        if self.current_zone_type == 'counting_lines':
            if len(self.current_points) >= 2:
                p1 = self.current_points[0]
                p2 = self.current_points[-1]
                direction = 'vertical' if self.direction_combo.currentText() == 'Vertical' else 'horizontal'
                if direction == 'horizontal':
                    line_pos = int((p1[1] + p2[1]) / 2)
                    x_start = int(min(p1[0], p2[0]))
                    x_end = int(max(p1[0], p2[0]))
                    if x_start == x_end:
                        x_start = 0
                        x_end = self.frame.shape[1] if self.frame is not None else x_start + 100
                    self.zones['counting_lines'].append({
                        'line_y': line_pos,
                        'direction': 'horizontal',
                        'x_start': x_start,
                        'x_end': x_end
                    })
                else:
                    line_pos = int((p1[0] + p2[0]) / 2)
                    y_start = int(min(p1[1], p2[1]))
                    y_end = int(max(p1[1], p2[1]))
                    if y_start == y_end:
                        y_start = 0
                        y_end = self.frame.shape[0] if self.frame is not None else y_start + 100
                    self.zones['counting_lines'].append({
                        'line_y': line_pos,
                        'direction': 'vertical',
                        'y_start': y_start,
                        'y_end': y_end
                    })
                self.current_points = []
                self._update_display()
                count = len(self.zones['counting_lines'])
                self.status_label.setText(f"[OK] Ligne {count} ajoutee!")
                self.info_label.setText(f"Lignes: {count}")
            else:
                self.status_label.setText("[!] Placez deux points pour définir la ligne")
        else:
            if len(self.current_points) >= 3:
                self.zones[self.current_zone_type].append(self.current_points.copy())
                self.current_points = []
                self._update_display()
                count = len(self.zones[self.current_zone_type])
                zone_name = self.zone_combo.currentText()
                self.status_label.setText(f"[OK] {zone_name} ajoutee!")
                self.info_label.setText(f"{zone_name}: {count}")
            else:
                self.status_label.setText("[!] Minimum 3 points requis")
    
    def _clear_all(self):
        """Clear all zones."""
        reply = QMessageBox.question(self, "Confirmation", "Effacer TOUTES les zones?")
        if reply == QMessageBox.StandardButton.Yes:
            self.zones = {
                'counting_lines': [],
                'forbidden_zones': [],
                'parking_zones': [],
                'loitering_zones': [],
                'restricted_areas': []
            }
            self.current_points = []
            self._update_display()
            self.status_label.setText("[DEL] Zones effacees")
            self.info_label.setText("")
    
    def _apply_zones(self):
        """Apply zones and close."""
        self.zones_modified.emit(self.zones)
    
    def _refresh_frame(self):
        """Refresh frame from callback."""
        if self.get_frame_callback:
            frame = self.get_frame_callback()
            if frame is not None:
                self.frame = frame
                self._update_display()
    
    def _update_display(self):
        """Update canvas display."""
        if self.frame is None:
            self.canvas.setText("Aucune image disponible")  # Mode texte si aucune frame n'est fournie
            return
        
        display = self.frame.copy()
        
        # Draw existing zones in semi-transparent
        self._draw_all_zones(display, alpha=0.3)
        
        # Draw current zone being drawn
        if self.current_points:
            color = (0, 255, 255)  # Cyan for current
            for i, pt in enumerate(self.current_points):
                cv2.circle(display, tuple([int(p) for p in pt]), 6, color, -1)
                cv2.circle(display, tuple([int(p) for p in pt]), 6, (255, 255, 255), 2)
            
            # Draw lines between points
            for i in range(len(self.current_points) - 1):
                p1 = tuple([int(p) for p in self.current_points[i]])
                p2 = tuple([int(p) for p in self.current_points[i + 1]])
                cv2.line(display, p1, p2, color, 2)
        
        # Convert to QPixmap
        h, w = display.shape[:2]
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        
        self.canvas.setPixmap(pixmap)
    
    def _draw_all_zones(self, display, alpha=0.3):
        """Draw all completed zones on display."""
        overlay = display.copy()
        
        # Draw counting lines
        for line in self.zones['counting_lines']:
            direction = line.get('direction', 'horizontal')
            if direction == 'vertical':
                x = int(line.get('line_y', display.shape[1] // 2))
                start_y = int(line.get('y_start', 0))
                end_y = line.get('y_end', display.shape[0])
                if end_y is None:
                    end_y = display.shape[0]
                try:
                    end_y = int(end_y)
                except Exception:
                    end_y = display.shape[0]
                if start_y >= end_y:
                    start_y, end_y = 0, display.shape[0]
                cv2.line(overlay, (x, start_y), (x, end_y), (0, 255, 0), 3)
            else:
                y = int(line.get('line_y', 0))
                start_x = int(line.get('x_start', 0))
                end_x = line.get('x_end', display.shape[1])
                if end_x is None:
                    end_x = display.shape[1]
                try:
                    end_x = int(end_x)
                except Exception:
                    end_x = display.shape[1]
                if start_x >= end_x:
                    start_x, end_x = 0, display.shape[1]
                cv2.line(overlay, (start_x, y), (end_x, y), (0, 255, 0), 3)
        
        # Draw other zones
        for zone_list, color in [
            (self.zones['parking_zones'], (255, 125, 0)),
            (self.zones['forbidden_zones'], (0, 0, 255)),
            (self.zones['loitering_zones'], (255, 0, 255)),
        ]:
            for zone in zone_list:
                points = np.array(zone, dtype=np.int32)
                cv2.polylines(overlay, [points], True, color, 2)
        
        # Blend
        cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)
    
    def _update_current_frame(self):
        """Get current frame (compatibility method)."""
        return self.frame
    
    def _current_frame(self):
        """Return current frame."""
        return self.frame


class ZoneCanvas(QWidget):
    """Canvas for zone drawing."""
    
    clicked = pyqtSignal(int, int)
    double_clicked = pyqtSignal(int, int)
    
    def __init__(self):
        super().__init__()
        self.pixmap = None
        self.text = "Cliquez pour ajouter des points"
        self.setMinimumHeight(400)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._scale = 1.0
        self._offset = (0, 0)
        self._drawn_size = (0, 0)
    
    def setPixmap(self, pixmap):
        """Set pixmap to display."""
        self.pixmap = pixmap
        self.update()
    
    def setText(self, text: str):
        """Show text message."""
        self.pixmap = None
        self.text = text
        self.update()
    
    def paintEvent(self, event):
        """Paint the canvas."""
        from PyQt6.QtGui import QPainter, QPen, QColor
        
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        if self.pixmap:
            scaled = self.pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            original_width = max(1, self.pixmap.width())
            self._scale = scaled.width() / original_width
            self._offset = (x, y)
            self._drawn_size = (scaled.width(), scaled.height())
        else:
            painter.setPen(QPen(QColor(100, 100, 100)))
            painter.setFont(QFont("Arial", 12))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text)
    
    def mousePressEvent(self, event):
        """Mouse clicked."""
        if self.pixmap:
            x = int(event.position().x())
            y = int(event.position().y())
            self.clicked.emit(x, y)
    
    def mouseDoubleClickEvent(self, event):
        """Double click."""
        if self.pixmap:
            x = int(event.position().x())
            y = int(event.position().y())
            self.double_clicked.emit(x, y)