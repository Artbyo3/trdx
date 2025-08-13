# 🎨 Assets - Recursos Gráficos de TRDX

Esta carpeta contiene todos los recursos gráficos del programa TRDX.

## 📁 Estructura de Carpetas

```
assets/
├── icons/                    # 🎯 Iconos del programa
│   ├── toolbar/             # Iconos para la barra de herramientas
│   │   ├── start.png        # Botón de iniciar
│   │   ├── stop.png         # Botón de detener
│   │   ├── camera.png       # Icono de cámara
│   │   └── settings.png     # Icono de configuración
│   └── status/              # Iconos de estado
│       ├── connected.png     # Estado conectado
│       ├── disconnected.png  # Estado desconectado
│       └── error.png        # Estado de error
├── splash/                   # 🖼️ Pantallas de carga
│   ├── splash_screen.png    # Pantalla de carga principal
│   ├── splash_screen.svg    # Versión vectorial
│   └── loading_animation.gif # Animación de carga
├── logos/                    # 🏷️ Logos y branding
│   ├── trdx_logo.png        # Logo principal TRDX
│   ├── trdx_logo.svg        # Versión vectorial del logo
│   └── trdx_banner.png      # Banner promocional
├── backgrounds/              # 🎨 Fondos y texturas
│   ├── dark_theme.png       # Fondo tema oscuro
│   ├── light_theme.png      # Fondo tema claro
│   └── gradient_bg.png      # Fondo con gradiente
└── ui/                       # 🖥️ Elementos de interfaz
    ├── buttons/              # Botones personalizados
    ├── panels/               # Paneles de interfaz
    └── overlays/             # Superposiciones y overlays
```

## 📋 Formatos Recomendados

### Iconos
- **Principal**: `.ico` (Windows), `.png` (otros sistemas)
- **Vectorial**: `.svg` (escalable)
- **Tamaños**: 16x16, 32x32, 48x48, 64x64, 128x128, 256x256 píxeles

### Pantallas de Carga
- **Formato**: `.png` o `.svg`
- **Tamaño**: 500x300 píxeles (recomendado)
- **Resolución**: 72 DPI para pantalla

### Logos
- **Principal**: `.png` con transparencia
- **Vectorial**: `.svg` para escalabilidad
- **Banner**: `.png` 1200x400 píxeles

## 🔧 Cómo Usar en el Código

### 1. Cargar Iconos
```python
from PySide6.QtGui import QIcon
from pathlib import Path

# Cargar icono principal
icon_path = Path(__file__).parent / "assets" / "icons" / "app_icon.png"
app.setWindowIcon(QIcon(str(icon_path)))
```

### 2. Cargar Pantalla de Carga
```python
from PySide6.QtGui import QPixmap

# Cargar splash screen
splash_path = Path(__file__).parent / "assets" / "splash" / "splash_screen.png"
pixmap = QPixmap(str(splash_path))
splash = QSplashScreen(pixmap)
```

### 3. Cargar Fondos
```python
# Cargar fondo
bg_path = Path(__file__).parent / "assets" / "backgrounds" / "dark_theme.png"
background = QPixmap(str(bg_path))
```

## 📝 Convenciones de Nomenclatura

- **Iconos**: `nombre_icono.png` (snake_case)
- **Pantallas**: `splash_screen.png`
- **Logos**: `trdx_logo.png`
- **Fondos**: `tema_nombre.png`

## 🎨 Especificaciones de Color

### Paleta TRDX
- **Azul Principal**: #64C8FF (100, 200, 255)
- **Azul Oscuro**: #4A90E2 (74, 144, 226)
- **Fondo Oscuro**: #1A1A2E (26, 26, 46)
- **Gris Claro**: #969696 (150, 150, 150)

## 📦 Archivos Necesarios

### Iconos Principales
- [ ] `icons/app_icon.ico` - Icono principal de la aplicación
- [ ] `icons/app_icon.png` - Versión PNG del icono
- [ ] `icons/app_icon.svg` - Versión vectorial

### Pantalla de Carga
- [ ] `splash/splash_screen.png` - Pantalla de carga actual
- [ ] `splash/splash_screen.svg` - Versión vectorial

### Logos
- [ ] `logos/trdx_logo.png` - Logo principal
- [ ] `logos/trdx_logo.svg` - Versión vectorial

### Iconos de Toolbar
- [ ] `icons/toolbar/start.png` - Botón iniciar
- [ ] `icons/toolbar/stop.png` - Botón detener
- [ ] `icons/toolbar/camera.png` - Icono cámara
- [ ] `icons/toolbar/settings.png` - Icono configuración

### Iconos de Estado
- [ ] `icons/status/connected.png` - Estado conectado
- [ ] `icons/status/disconnected.png` - Estado desconectado
- [ ] `icons/status/error.png` - Estado error

## 🚀 Próximos Pasos

1. **Crear iconos** en los tamaños recomendados
2. **Diseñar pantalla de carga** con el logo TRDX
3. **Implementar carga de recursos** en el código
4. **Crear iconos de toolbar** para la interfaz
5. **Diseñar fondos** para diferentes temas

## 📞 Soporte

Para agregar nuevos recursos gráficos:
1. Coloca el archivo en la carpeta correspondiente
2. Actualiza este README si es necesario
3. Modifica el código para cargar el nuevo recurso 