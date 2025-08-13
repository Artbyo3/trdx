# TRDX Driver - Changelog

## Versión 1.1.0 - Optimización de Precisión (Agosto 2025)

### Cambios en el Driver (trdx_driver.cpp)

#### Filtrado Reducido para Máxima Precisión
- **Antes**: Filtrado 70% datos nuevos, 30% filtrado
- **Ahora**: Filtrado 95% datos nuevos, 5% filtrado
- **Impacto**: Movimientos mucho más precisos y responsivos

#### Tamaño de Filtro Reducido
- **Antes**: FILTER_SIZE = 3
- **Ahora**: FILTER_SIZE = 2
- **Impacto**: Menor latencia en el procesamiento

### Cambios en el Código Python

#### Transformaciones de Coordenadas Centralizadas
- Nueva función `_apply_coordinate_transformations()` en `pose_processor.py`
- Eliminación de transformaciones duplicadas en `mediapipepose.py`
- **Impacto**: Consistencia en las transformaciones de coordenadas

#### Filtrado Mejorado
- **Antes**: 80% datos nuevos, 20% filtrado
- **Ahora**: 90% datos nuevos, 10% filtrado
- **Impacto**: Preservación de la precisión de MediaPipe

#### Optimización UDP
- **Antes**: 10 reintentos, 0.1s espera
- **Ahora**: 5 reintentos, 0.05s espera
- **Impacto**: Menor latencia en la comunicación

#### Mapeo de Landmarks Optimizado
- Comentarios mejorados para usar landmarks más precisos
- **Impacto**: Mejor precisión en la detección de poses

### Nuevas Características

#### Monitoreo de Precisión
- Nuevas métricas en `performance_monitor.py`:
  - `tracker_precision`: Precisión por tracker
  - `coordinate_consistency`: Consistencia de transformaciones
  - `mediapipe_confidence`: Confianza promedio de MediaPipe

#### Configuración de Precisión
- Nuevas variables en `pose_processor.py`:
  - `precision_mode`: Modo de máxima precisión
  - `smoothing_factor`: Factor de suavizado reducido
  - `coordinate_scale`: Escala de coordenadas

### Instalación

1. **Compilar el driver**:
   ```bash
   cd steamvr_driver
   .\build_driver.bat
   ```

2. **Instalar el driver**:
   ```bash
   .\install_driver.bat
   ```

3. **Reiniciar SteamVR** completamente

### Resultados Esperados

- **Movimientos más precisos**: Los trackers deberían replicar exactamente los movimientos de MediaPipe
- **Menor latencia**: Respuesta más rápida a los movimientos
- **Mejor separación de pies**: Los trackers de pies deberían moverse de forma independiente
- **Consistencia mejorada**: Menos saltos o movimientos erráticos

### Notas Importantes

- Asegúrate de que SteamVR esté completamente cerrado antes de instalar
- Si los trackers no aparecen, reinicia SteamVR completamente
- Los cambios son compatibles con versiones anteriores del código Python 