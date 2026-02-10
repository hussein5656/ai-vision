"""Dark theme and shared UI components."""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QFileDialog,
                             QComboBox, QMessageBox)
from PyQt6.QtCore import Qt


DARK_STYLESHEET = """
    QWidget {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    QLabel {
        color: #ffffff;
    }
    QPushButton {
        background-color: #0d47a1;
        color: white;
        border: none;
        padding: 5px;
        border-radius: 4px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #1565c0;
    }
    QPushButton:pressed {
        background-color: #0d47a1;
    }
    QPushButton#startBtn {
        background-color: #2e7d32;
    }
    QPushButton#startBtn:hover {
        background-color: #388e3c;
    }
    QPushButton#stopBtn {
        background-color: #c62828;
    }
    QPushButton#stopBtn:hover {
        background-color: #d32f2f;
    }
    QPushButton#zoneBtn {
        background-color: #f57c00;
    }
    QPushButton#zoneBtn:hover {
        background-color: #e65100;
    }
    QLineEdit {
        background-color: #2d2d2d;
        color: #ffffff;
        border: 1px solid #0d47a1;
        border-radius: 4px;
        padding: 5px;
    }
    QComboBox {
        background-color: #2d2d2d;
        color: #ffffff;
        border: 1px solid #0d47a1;
        border-radius: 4px;
        padding: 5px;
    }
    QComboBox QAbstractItemView {
        background-color: #2d2d2d;
        color: #ffffff;
        selection-background-color: #0d47a1;
    }
    QTabWidget::pane {
        border: 1px solid #0d47a1;
    }
    QTabBar::tab {
        background-color: #2d2d2d;
        color: #ffffff;
        padding: 8px 20px;
        border: 1px solid #0d47a1;
    }
    QTabBar::tab:selected {
        background-color: #0d47a1;
        color: #ffffff;
    }
    QTableWidget {
        background-color: #2d2d2d;
        color: #ffffff;
        gridline-color: #0d47a1;
    }
    QTableWidget::item {
        padding: 5px;
        border: none;
    }
    QHeaderView::section {
        background-color: #1a1a1a;
        color: #ffffff;
        padding: 5px;
        border: 1px solid #0d47a1;
    }
    QCheckBox {
        color: #ffffff;
    }
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
    }
    QCheckBox::indicator:unchecked {
        background-color: #2d2d2d;
        border: 1px solid #0d47a1;
    }
    QCheckBox::indicator:checked {
        background-color: #0d47a1;
        border: 1px solid #0d47a1;
    }
    QGroupBox {
        color: #ffffff;
        border: 2px solid #0d47a1;
        border-radius: 4px;
        margin-top: 10px;
        padding-top: 10px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 3px 0 3px;
    }
    QDialog {
        background-color: #1e1e1e;
    }
    QMessageBox {
        background-color: #1e1e1e;
    }
    QMessageBox QLabel {
        color: #ffffff;
    }
    QSpinBox {
        background-color: #2d2d2d;
        color: #ffffff;
        border: 1px solid #0d47a1;
        border-radius: 4px;
        padding: 5px;
    }
"""


class SourceDialog(QDialog):
    """Dialog for adding new video sources."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ajouter une source vidéo")
        self.setGeometry(200, 200, 500, 350)
        self.setStyleSheet(DARK_STYLESHEET)
        self.source = None
        self.source_type = None
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout()
        
        # Source type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type de source:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems([
            "Caméra USB",
            "Flux RTSP/HTTP",
            "Fichier vidéo",
            "Dossier d'images"
        ])
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)
        
        # Input fields
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Source:"))
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Numéro de caméra (0, 1, 2...) ou URL ou chemin")
        input_layout.addWidget(self.input_field)
        
        browse_btn = QPushButton("Parcourir")
        browse_btn.clicked.connect(self._browse)
        input_layout.addWidget(browse_btn)
        layout.addLayout(input_layout)
        
        # Help text
        help_text = QLabel(
            "Caméra USB: entrez le numéro (0, 1, 2...)\n"
            "RTSP/HTTP: entrez l'URL complète\n"
            "Vidéo: sélectionnez un fichier .mp4, .avi, etc.\n"
            "Images: sélectionnez un dossier"
        )
        help_text.setStyleSheet("color: #888888; font-size: 10px;")
        layout.addWidget(help_text)
        
        # Buttons
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self._validate_and_accept)
        cancel_btn = QPushButton("Annuler")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
    
    def _on_type_changed(self, index):
        """Type changed."""
        placeholders = [
            "Numéro de caméra (0, 1, 2...)",
            "rtsp://example.com/stream ou http://...",
            "Chemin du fichier vidéo",
            "Chemin du dossier d'images"
        ]
        self.input_field.setPlaceholderText(placeholders[index])
    
    def _browse(self):
        """Browse for file or folder."""
        source_type = self.type_combo.currentIndex()
        
        if source_type == 2:  # Video file
            path, _ = QFileDialog.getOpenFileName(
                self, "Sélectionner vidéo",
                filter="Vidéos (*.mp4 *.avi *.mov *.mkv *.flv);;Tous (*.*)"
            )
            if path:
                self.input_field.setText(path)
        
        elif source_type == 3:  # Images folder
            path = QFileDialog.getExistingDirectory(self, "Sélectionner dossier d'images")
            if path:
                self.input_field.setText(path)
    
    def _validate_and_accept(self):
        """Validate input and accept."""
        text = self.input_field.text().strip()
        if not text:
            QMessageBox.warning(self, "Erreur", "Veuillez entrer une source")
            return
        
        source_type = self.type_combo.currentIndex()
        
        if source_type == 0:  # USB camera
            try:
                self.source = int(text)
            except:
                QMessageBox.warning(self, "Erreur", "Entrez un numéro de caméra (0, 1, 2...)")
                return
        else:  # URL, video, or folder
            import os
            if source_type == 2:  # video file
                if not os.path.isfile(text):
                    QMessageBox.warning(self, "Erreur", "Fichier vidéo introuvable")
                    return
            if source_type == 3:  # images folder
                if not os.path.isdir(text):
                    QMessageBox.warning(self, "Erreur", "Dossier d'images introuvable")
                    return
            self.source = text
        
        self.source_type = source_type
        self.accept()