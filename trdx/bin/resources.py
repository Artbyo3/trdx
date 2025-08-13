"""
Resources Helper - Carga centralizada de recursos gráficos
Facilita el acceso a iconos, imágenes y otros recursos gráficos
"""

import os
from pathlib import Path
from typing import Optional
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtCore import QSize

class ResourceManager:
    """Gestor centralizado de recursos gráficos"""
    
    def __init__(self):
        # Obtener ruta base del proyecto
        self.base_path = Path(__file__).parent.parent
        self.assets_path = self.base_path / "assets"
        
        # Verificar que existe la carpeta assets
        if not self.assets_path.exists():
            raise FileNotFoundError(f"Assets folder not found: {self.assets_path}")
    
    def get_asset_path(self, *path_parts: str) -> Path:
        """Obtener ruta completa a un recurso"""
        return self.assets_path.joinpath(*path_parts)
    
    def load_icon(self, icon_name: str, size: Optional[QSize] = None) -> QIcon:
        """Cargar icono desde assets/icons/"""
        icon_path = self.get_asset_path("icons", icon_name)
        
        if not icon_path.exists():
            # Buscar en subcarpetas
            for subfolder in ["toolbar", "status"]:
                subfolder_path = self.get_asset_path("icons", subfolder, icon_name)
                if subfolder_path.exists():
                    icon_path = subfolder_path
                    break
        
        if not icon_path.exists():
            raise FileNotFoundError(f"Icon not found: {icon_name}")
        
        icon = QIcon(str(icon_path))
        
        if size:
            icon = QIcon(icon.pixmap(size))
        
        return icon
    
    def load_pixmap(self, image_name: str, size: Optional[QSize] = None) -> QPixmap:
        """Cargar imagen como QPixmap"""
        image_path = self.get_asset_path(image_name)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_name}")
        
        pixmap = QPixmap(str(image_path))
        
        if size and not pixmap.isNull():
            pixmap = pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        return pixmap
    
    def load_mascot(self) -> QPixmap:
        """Cargar imagen de la mascota específicamente"""
        try:
            return self.load_pixmap("splash/loading.png")
        except FileNotFoundError:
            # Crear una mascota por defecto si no existe
            return self._create_default_mascot()
    
    def _create_default_mascot(self) -> QPixmap:
        """Crear una mascota por defecto si no existe la imagen"""
        from PySide6.QtGui import QPainter, QColor, QFont
        from PySide6.QtCore import Qt
        
        # Crear pixmap 120x120 para la mascota
        pixmap = QPixmap(120, 120)
        pixmap.fill(QColor(0, 0, 0, 0))  # Transparente
        
        # Dibujar una mascota simple por defecto
        painter = QPainter(pixmap)
        painter.setPen(QColor(100, 200, 100))  # Verde claro
        painter.setBrush(QColor(100, 200, 100))
        
        # Dibujar un círculo simple como mascota
        painter.drawEllipse(20, 20, 80, 80)
        
        # Dibujar ojos
        painter.setPen(QColor(255, 255, 255))
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(35, 35, 15, 15)
        painter.drawEllipse(70, 35, 15, 15)
        
        # Pupilas
        painter.setPen(QColor(0, 0, 0))
        painter.setBrush(QColor(0, 0, 0))
        painter.drawEllipse(38, 38, 9, 9)
        painter.drawEllipse(73, 38, 9, 9)
        
        painter.end()
        return pixmap
    
    def load_splash_screen(self, splash_name: str = "splash_screen.png") -> QPixmap:
        """Cargar pantalla de carga"""
        return self.load_pixmap(f"splash/{splash_name}")
    
    def load_logo(self, logo_name: str = "trdx_logo.png") -> QPixmap:
        """Cargar logo"""
        return self.load_pixmap(f"logos/{logo_name}")
    
    def load_background(self, bg_name: str) -> QPixmap:
        """Cargar fondo"""
        return self.load_pixmap(f"backgrounds/{bg_name}")
    
    def get_app_icon(self) -> QIcon:
        """Obtener icono principal de la aplicación"""
        # Intentar cargar .ico primero (Windows)
        try:
            return self.load_icon("app_icon.ico")
        except FileNotFoundError:
            # Fallback a PNG
            try:
                return self.load_icon("app_icon.png")
            except FileNotFoundError:
                # Fallback a SVG
                try:
                    return self.load_icon("app_icon.svg")
                except FileNotFoundError:
                    # Crear icono por defecto
                    return self._create_default_icon()
    
    def _create_default_icon(self) -> QIcon:
        """Crear icono por defecto si no existe"""
        from PySide6.QtGui import QPainter, QColor, QFont
        from PySide6.QtCore import Qt
        
        # Crear pixmap 256x256
        pixmap = QPixmap(256, 256)
        pixmap.fill(QColor(25, 25, 35))  # Fondo oscuro
        
        # Dibujar "T" simple
        painter = QPainter(pixmap)
        painter.setPen(QColor(100, 200, 255))  # Azul TRDX
        font = QFont("Arial", 120, QFont.Bold)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "T")
        painter.end()
        
        return QIcon(pixmap)
    
    def list_available_icons(self) -> list:
        """Listar todos los iconos disponibles"""
        icons = []
        icons_path = self.get_asset_path("icons")
        
        if icons_path.exists():
            for file_path in icons_path.rglob("*.png"):
                icons.append(str(file_path.relative_to(icons_path)))
            for file_path in icons_path.rglob("*.ico"):
                icons.append(str(file_path.relative_to(icons_path)))
            for file_path in icons_path.rglob("*.svg"):
                icons.append(str(file_path.relative_to(icons_path)))
        
        return sorted(icons)
    
    def list_available_splashes(self) -> list:
        """Listar todas las pantallas de carga disponibles"""
        splashes = []
        splash_path = self.get_asset_path("splash")
        
        if splash_path.exists():
            for file_path in splash_path.glob("*"):
                if file_path.is_file():
                    splashes.append(file_path.name)
        
        return sorted(splashes)

# Instancia global del gestor de recursos
resource_manager = ResourceManager()

# Funciones de conveniencia
def get_icon(icon_name: str, size: Optional[QSize] = None) -> QIcon:
    """Cargar icono"""
    return resource_manager.load_icon(icon_name, size)

def get_pixmap(image_name: str, size: Optional[QSize] = None) -> QPixmap:
    """Cargar imagen como QPixmap"""
    return resource_manager.load_pixmap(image_name, size)

def get_splash_screen(splash_name: str = "splash_screen.png") -> QPixmap:
    """Cargar pantalla de carga"""
    return resource_manager.load_splash_screen(splash_name)

def get_logo(logo_name: str = "trdx_logo.png") -> QPixmap:
    """Cargar logo"""
    return resource_manager.load_logo(logo_name)

def get_app_icon() -> QIcon:
    """Obtener icono principal de la aplicación"""
    return resource_manager.get_app_icon()

def get_mascot() -> QPixmap:
    """Obtener imagen de la mascota"""
    return resource_manager.load_mascot() 