# Proyecto PDS 2024-25: Aplicación de Verificación de locutores

Este proyecto implementa un sistema básico de verificación de locutores mediante biometría de voz. Consta de:
- **Cliente web** (`cliente_web.html`): Permite grabar y enviar audios para registrar o verificar un usuario.
- **Servidor** (`server.py`): Procesa los audios y compara características de voz.

---
∫
## 🚀 Instalación y ejecución

### 1️⃣ Requisitos previos
Asegúrate de tener instalado:
- **Python 3.7+**
- **Visual Studio Code (VSCode)**

### 2️⃣ Configurar y ejecutar el servidor
1. Abre VSCode y carga la carpeta del proyecto.
2. Instala las dependencias ejecutando en la terminal:
   ```sh
   pip install flask flask_cors librosa numpy scipy
   ```
3. Ejecuta el servidor:
   ```sh
   python server.py
   ```
4. Si el servidor se ejecuta correctamente, deberías ver:
   ```
   * Running on http://127.0.0.1:5000/
   ```

### 3️⃣ Ejecutar el cliente web
1. Abre `static/index.html` en tu navegador (doble clic o arrástralo a la ventana del navegador) o, de forma alternativa, ve con el navegador a la siguiente URL: [http://127.0.0.1:5000](http://127.0.0.1:5000).
2. Ingresa un **ID de usuario** (p. ej. 0).
3. Usa los botones para **grabar y enviar audio** al servidor.
4. El resultado de la verificación aparecerá en pantalla.

---

## 🛠 Solución de problemas
❌ **Error de conexión en la web**: Asegúrate de que el servidor está corriendo en `http://127.0.0.1:5000/` y que el Firewall no te bloquea dicho puerto.

❌ **No se graba audio**: Revisa los permisos del micrófono en tu navegador.

❌ **Error en Flask**: Verifica que tienes Python y las librerías instaladas.

❌ **Problemas al instalar las librerias**: Prueba a ejecutar el software un entorno virtual con una versión anterior de Python:
   ```
   conda create --name proyecto_pds_2025 python=3.11
   conda activate proyecto_pds_2025
   conda install flask flask_cors librosa numpy scipy
   ```
---

