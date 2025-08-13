"""
TRDX - Punto de entrada principal REFACTORIZADO
Ejecuta la aplicación de full body tracking con event loop seguro
"""

import sys
import time
import logging
import threading
from pathlib import Path
from PySide6.QtWidgets import QApplication, QSplashScreen, QDialog, QWidget, QVBoxLayout, QHBoxLayout, QProgressBar, QLabel
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QFont, QPainter, QColor, QLinearGradient, QIcon

class SplashDialog(QDialog):
    """Splash screen mejorado con progress bar"""
    
    def __init__(self):
        super().__init__()
        self.setFixedSize(500, 350)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Centrar en pantalla
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)
        
        # Widget principal
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # Layout principal
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(main_widget)
        self.layout().setContentsMargins(0, 0, 0, 0)
        
        # Fondo
        main_widget.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                border-radius: 10px;
            }
        """)
        
        # Área para mascota
        mascot_label = QLabel()
        mascot_label.setFixedSize(140, 140)
        mascot_label.setAlignment(Qt.AlignCenter)
        
        # Cargar mascota
        try:
            from bin.resources import get_mascot
            mascot_pixmap = get_mascot()
            if not mascot_pixmap.isNull():
                scaled_mascot = mascot_pixmap.scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                mascot_label.setPixmap(scaled_mascot)
        except Exception:
            mascot_label.setText("🦎")
            mascot_label.setStyleSheet("font-size: 60px; color: #ffffff;")
        
        # Contenedor para centrar mascota
        mascot_container = QWidget()
        mascot_layout = QHBoxLayout(mascot_container)
        mascot_layout.addStretch()
        mascot_layout.addWidget(mascot_label)
        mascot_layout.addStretch()
        
        # Logo TRDX
        logo_label = QLabel("TRDX")
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 32px;
                font-weight: bold;
            }
        """)
        
        # Subtítulo
        subtitle_label = QLabel("Multi-Camera Full Body Tracking")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 14px;
            }
        """)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #444444;
                border-radius: 5px;
                text-align: center;
                background-color: #333333;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        # Texto de estado
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 14px;
            }
        """)
        
        # Agregar elementos
        main_layout.addWidget(mascot_container)
        main_layout.addWidget(logo_label)
        main_layout.addWidget(subtitle_label)
        main_layout.addStretch()
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.status_label)
        
        # Timer para timeout - DESHABILITADO porque la aplicación funciona correctamente
        # self.timeout_timer = QTimer()
        # self.timeout_timer.timeout.connect(self.handle_timeout)
        # self.timeout_timer.setSingleShot(True)
        # self.timeout_timer.start(30000)  # 30 segundos
        
    def update_progress(self, value, message):
        """Actualizar progress bar y mensaje"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        QApplication.processEvents()
        
    def show_error(self, error_message):
        """Mostrar error en splash"""
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #FF6464;
                font-size: 14px;
            }
        """)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #FF6464;
                border-radius: 5px;
                text-align: center;
                background-color: #333333;
            }
            QProgressBar::chunk {
                background-color: #FF6464;
                border-radius: 3px;
            }
        """)
        QApplication.processEvents()
        
    def handle_timeout(self):
        """Manejar timeout"""
        self.show_error("Timeout - Application taking too long to start")
        print("❌ TIMEOUT: La aplicación está tardando demasiado en iniciar")
        QTimer.singleShot(5000, self.close)
        QTimer.singleShot(5000, lambda: sys.exit(1))

def main():
    """Función principal refactorizada - SIMPLIFICADA"""
    # Crear aplicación Qt
    app = QApplication(sys.argv)
    app.setApplicationName("TRDX")
    
    # Configurar icono
    try:
        icon_path = Path(__file__).parent / "assets" / "logos" / "logo.ico"
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
            print(f"✅ Icono cargado: {icon_path}")
        else:
            print(f"⚠️ Icono no encontrado en: {icon_path}")
    except Exception as e:
        print(f"⚠️ Error al cargar icono: {e}")
    
    # Crear splash screen
    splash = SplashDialog()
    splash.show()
    app.processEvents()
    
    def initialize_application():
        """Inicializar aplicación de forma segura"""
        try:
            splash.update_progress(20, "Loading modules...")
            
            # Importar aplicación
            from bin.trdx_app import TRDXApplication
            
            splash.update_progress(40, "Creating application instance...")
            
            # Crear instancia de aplicación
            app_instance = TRDXApplication()
            app_instance.app = app
            
            splash.update_progress(60, "Initializing components...")
            
            # Inicializar en el thread principal
            if app_instance.initialize_in_main_thread():
                splash.update_progress(100, "Ready!")
                print("✅ Aplicación inicializada correctamente")
                
                # Debug: Verificar que la ventana se creó
                if hasattr(app_instance, 'main_window') and app_instance.main_window:
                    print("🔍 Debug: Main window creada correctamente")
                else:
                    print("❌ Error: Main window no se creó")
                
                # Cerrar splash y mostrar ventana principal
                QTimer.singleShot(1000, lambda: show_main_window(app_instance))
            else:
                splash.show_error("Failed to initialize application")
                print("❌ Error de inicialización")
                QTimer.singleShot(5000, lambda: sys.exit(1))
                
        except ImportError as e:
            splash.show_error(f"Module import error: {e}")
            print(f"❌ Error de importación: {e}")
            QTimer.singleShot(5000, lambda: sys.exit(1))
        except Exception as e:
            splash.show_error(f"Initialization error: {e}")
            print(f"❌ Error de inicialización: {e}")
            QTimer.singleShot(5000, lambda: sys.exit(1))
    
    def show_main_window(app_instance):
        """Mostrar ventana principal"""
        try:
            print("🔍 Debug: Cerrando splash screen...")
            splash.close()
            
            print("🔍 Debug: Verificando main_window...")
            if hasattr(app_instance, 'main_window') and app_instance.main_window:
                # IMPORTANTE: Mantener referencia global a la ventana
                global main_window_ref
                main_window_ref = app_instance.main_window
                
                # Asegurar que la ventana sea visible y esté en primer plano
                main_window_ref.show()
                main_window_ref.raise_()
                main_window_ref.activateWindow()
                
                # Prevenir que la ventana se cierre automáticamente
                main_window_ref.setAttribute(Qt.WA_DeleteOnClose, False)
                
                # Conectar señal de cierre para debugging
                def on_window_closing():
                    print("⚠️ Ventana principal se está cerrando - esto no debería pasar automáticamente")
                
                main_window_ref.destroyed.connect(on_window_closing)
                
                # Centrar la ventana en la pantalla
                screen = QApplication.primaryScreen().geometry()
                window_rect = main_window_ref.geometry()
                x = (screen.width() - window_rect.width()) // 2
                y = (screen.height() - window_rect.height()) // 2
                main_window_ref.move(x, y)
                
                # Forzar el foco en la ventana
                main_window_ref.setFocus()
                main_window_ref.setWindowState(Qt.WindowActive)
                
                print("💡 La aplicación está lista para usar")
                print("🎯 Busca la ventana 'TRDX - Multi-Camera Full Body Tracking' en tu pantalla")
                print("📌 Si no la ves, revisa la barra de tareas o presiona Alt+Tab")
                
                # Verificar estado después de 2 segundos
                def check_window_state():
                    if main_window_ref and main_window_ref.isVisible():
                        print("✅ Ventana visible correctamente")
                        print("🎉 ¡TRDX está listo para usar!")
                    else:
                        print("❌ Ventana no visible - intentando mostrar nuevamente")
                        if main_window_ref:
                            main_window_ref.show()
                            main_window_ref.raise_()
                
                QTimer.singleShot(2000, check_window_state)
                
                # Timer para verificar que la ventana siga existiendo
                def check_window_exists():
                    if main_window_ref and main_window_ref.isVisible():
                        print("✅ Ventana sigue funcionando")
                    else:
                        print("⚠️ Ventana no visible - restaurando...")
                        if main_window_ref:
                            main_window_ref.show()
                            main_window_ref.raise_()
                        else:
                            print("❌ CRÍTICO: Main window se perdió completamente")
                
                window_check_timer = QTimer()
                window_check_timer.timeout.connect(check_window_exists)
                window_check_timer.start(10000)  # Verificar cada 10 segundos
                
                # IMPORTANTE: Mantener referencia a la aplicación para evitar que se cierre
                app_instance._keep_alive = True
                
            else:
                print("❌ Error: main_window no existe o es None")
                splash.show_error("Main window not found")
                QTimer.singleShot(5000, lambda: sys.exit(1))
                
        except Exception as e:
            splash.show_error(f"Error showing main window: {e}")
            print(f"❌ Error mostrando ventana principal: {e}")
            import traceback
            print(f"🔍 Debug: Traceback completo: {traceback.format_exc()}")
            QTimer.singleShot(5000, lambda: sys.exit(1))
    
    # Iniciar inicialización después de mostrar splash
    QTimer.singleShot(100, initialize_application)
    
    # Timer para mantener la aplicación viva y detectar problemas
    keep_alive_timer = QTimer()
    keep_alive_timer.timeout.connect(lambda: check_app_status())
    keep_alive_timer.start(5000)  # Verificar cada 5 segundos
    
    def check_app_status():
        """Verificar que la aplicación siga funcionando"""
        try:
            if hasattr(app, 'activeWindow') and app.activeWindow():
                print("✅ Aplicación funcionando correctamente")
            else:
                print("⚠️ No hay ventana activa - verificando estado...")
        except Exception as e:
            print(f"❌ Error verificando estado de la aplicación: {e}")
    
    # Ejecutar aplicación
    try:
        print("🚀 Iniciando event loop de la aplicación...")
        exit_code = app.exec()
        print(f"📤 Aplicación terminada con código: {exit_code}")
        return exit_code
    except Exception as e:
        print(f"❌ Error en aplicación: {e}")
        import traceback
        print(f"🔍 Debug: Traceback completo: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    main()
