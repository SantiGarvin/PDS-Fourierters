from flask import Flask, request, jsonify, send_file
from flask_cors import CORS # Importar Flask-CORS
import os
from glob import iglob
from pathlib import Path
import asv # Importa nuestro módulo asv modificado
import json
import logging
import numpy as np
import tempfile
import shutil

# --- Configuración ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# --- Constantes y Rutas ---
DATASET_FOLDER = "audios"
FINGERPRINT_DB_FILE = "fingerprints.db.json" # Archivo para persistencia
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Diccionario donde almacenaremos las huellas vocales de los usuarios (se carga/guarda)
reg_user_vocal_fingerprint = dict()

# --- Funciones Auxiliares para Persistencia ---
def load_fingerprints():
    """Carga las huellas vocales desde el archivo JSON."""
    global reg_user_vocal_fingerprint
    try:
        if os.path.exists(FINGERPRINT_DB_FILE):
            with open(FINGERPRINT_DB_FILE, 'r') as f:
                data_serializable = json.load(f)
                # Convertir las listas de vuelta a numpy arrays
                reg_user_vocal_fingerprint = {
                    user_id: np.array(fp_list)
                    for user_id, fp_list in data_serializable.items()
                }
                logging.info(f"Cargadas {len(reg_user_vocal_fingerprint)} huellas desde {FINGERPRINT_DB_FILE}")
        else:
            logging.info(f"Archivo {FINGERPRINT_DB_FILE} no encontrado. Iniciando con registro vacío.")
            reg_user_vocal_fingerprint = {}
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Error al cargar {FINGERPRINT_DB_FILE}: {e}. Iniciando con registro vacío.")
        reg_user_vocal_fingerprint = {} # Empezar vacío en caso de error
    except Exception as e: # Captura general para problemas con np.array si los datos son corruptos
        logging.error(f"Error inesperado al procesar huellas desde {FINGERPRINT_DB_FILE}: {e}. Iniciando con registro vacío.")
        reg_user_vocal_fingerprint = {}


def save_fingerprints():
    """Guarda el diccionario de huellas vocales en el archivo JSON."""
    try:
        # Convertir numpy arrays a listas para que sean serializables en JSON
        data_serializable = {
            user_id: fp.tolist()
            for user_id, fp in reg_user_vocal_fingerprint.items()
        }
        with open(FINGERPRINT_DB_FILE, 'w') as f:
            json.dump(data_serializable, f, indent=4) # indent=4 para que sea legible
        logging.info(f"Huellas guardadas en {FINGERPRINT_DB_FILE}")
    except IOError as e:
        logging.error(f"Error al guardar huellas en {FINGERPRINT_DB_FILE}: {e}")
    except TypeError as e:
         logging.error(f"Error de tipo al serializar huellas a JSON: {e}") # e.g., si algo no es numpy array

# --- Carga inicial de huellas ---
load_fingerprints()


@app.route("/")
def home_page():
    # Servir index.html desde la carpeta static
    return send_file('static/index.html')

# Registro de voz
@app.route("/register", methods=["POST"])
def register():
    user_id = request.form.get("user_id")
    if not user_id:
        return jsonify({"error": "Falta el ID del usuario"}), 400

    # Validar formato de user_id (ej: no vacío, sin caracteres extraños)
    user_id = user_id.strip() # Limpiar espacios
    if not user_id or not user_id.isalnum(): # Permitir solo alfanuméricos (o ajustar según necesidad)
         return jsonify({"error": "ID de usuario inválido (solo letras y números permitidos)"}), 400

    if user_id in reg_user_vocal_fingerprint:
        return jsonify({"error": f"Usuario '{user_id}' ya registrado"}), 409

    file = request.files.get("audio")
    if not file or not file.filename: # Comprobar también que tenga nombre
        return jsonify({"error": "No se proporcionó un archivo de audio válido"}), 400

    # Usar un directorio temporal para el archivo de registro
    temp_dir = None # Inicializar fuera del try
    try:
        temp_dir = tempfile.TemporaryDirectory()
        temp_audio_path = os.path.join(temp_dir.name, f"register_{user_id}.wav") # Nombre temporal único
        file.save(temp_audio_path)
        logging.info(f"Audio de registro temporal guardado en: {temp_audio_path}")

        # Calculamos su huella vocal usando la función actualizada de asv.py
        new_fingerprint = asv.compute_vocal_fingerprint(temp_audio_path)

        if new_fingerprint is None:
             # compute_vocal_fingerprint ya logueó el error específico
             return jsonify({"error": "Error al procesar el audio (puede ser muy corto o inválido)"}), 500

        # Si el cálculo fue exitoso, guardar la huella y persistir
        reg_user_vocal_fingerprint[user_id] = new_fingerprint
        logging.info(f'Registrando a: {user_id} con huella de tamaño {new_fingerprint.shape}')

        # Guardar el diccionario actualizado en el archivo JSON
        save_fingerprints()

        # Guardar el audio permanentemente en 'audios' DESPUÉS de un registro exitoso
        permanent_audio_path = os.path.join(DATASET_FOLDER, f"{user_id}.wav")
        shutil.copyfile(temp_audio_path, permanent_audio_path)
        logging.info(f"Audio de registro copiado a: {permanent_audio_path}")

        # El directorio temporal se limpiará automáticamente al salir del bloque 'with' o 'try...finally'

        return jsonify({"message": f"Usuario '{user_id}' registrado correctamente."})

    except Exception as e:
        logging.error(f"Error inesperado durante el registro de {user_id}: {e}", exc_info=True) # Log stack trace
        # Asegurarse de que el usuario no quede registrado si hubo un error
        if user_id in reg_user_vocal_fingerprint:
            del reg_user_vocal_fingerprint[user_id]
            # Podríamos intentar guardar el estado sin el usuario fallido, pero puede ser complejo
        return jsonify({"error": "Error interno del servidor durante el registro"}), 500
    finally:
        # Asegurar la limpieza del directorio temporal si se creó
        if temp_dir:
            try:
                 temp_dir.cleanup()
                 logging.info(f"Directorio temporal {temp_dir.name} limpiado.")
            except Exception as e:
                 logging.error(f"Error limpiando directorio temporal {temp_dir.name}: {e}")


# Verificación de locutor
@app.route("/verify", methods=["POST"])
def verify():
    user_id = request.form.get("user_id")
    if not user_id:
        return jsonify({"error": "Falta el ID del usuario"}), 400

    user_id = user_id.strip()
    if not user_id:
         return jsonify({"error": "ID de usuario inválido"}), 400

    file = request.files.get("audio")
    if not file or not file.filename:
        return jsonify({"error": "No se proporcionó un archivo de audio válido"}), 400

    if user_id not in reg_user_vocal_fingerprint:
        return jsonify({"error": f"El usuario '{user_id}' no está registrado"}), 404

    logging.info(f'Verificando a: {user_id}')

    registered_fingerprint = reg_user_vocal_fingerprint[user_id]

    temp_dir = None
    try:
        temp_dir = tempfile.TemporaryDirectory()
        test_audio_path = os.path.join(temp_dir.name, f"verify_{user_id}_temp.wav")
        file.save(test_audio_path)
        logging.info(f"Audio de verificación guardado temporalmente en: {test_audio_path}")

        # Calcular la huella vocal de la nueva muestra de audio
        test_fingerprint = asv.compute_vocal_fingerprint(test_audio_path)

        if test_fingerprint is None:
            # compute_vocal_fingerprint ya logueó el error
             return jsonify({"error": "Error al procesar el audio de verificación"}), 500

        # Comparamos ambas huellas vocales usando la distancia Coseno
        # Usamos el umbral definido en asv.py (COSINE_THRESHOLD)
        verified, distance = asv.compare_vocal_fingerprints(registered_fingerprint, test_fingerprint)

        # Asegurarse de que los tipos son serializables para JSON
        verified_bool = bool(verified)
        # Manejar distancia infinita si la comparación falló internamente
        distance_float = float('inf') if np.isinf(distance) else float(distance)

        logging.info(f'Usuario: {user_id} - Verificado: {verified_bool} - Distancia Coseno: {distance_float:.4f} (Umbral: {asv.COSINE_THRESHOLD})')

        return jsonify({"verified": verified_bool, "distance": distance_float})

    except Exception as e:
        logging.error(f"Error inesperado durante la verificación de {user_id}: {e}", exc_info=True)
        return jsonify({"error": "Error interno del servidor durante la verificación"}), 500
    finally:
        # Asegurar limpieza del directorio temporal
        if temp_dir:
            try:
                 temp_dir.cleanup()
                 logging.info(f"Directorio temporal de verificación {temp_dir.name} limpiado.")
            except Exception as e:
                 logging.error(f"Error limpiando directorio temporal de verificación {temp_dir.name}: {e}")


if __name__ == "__main__":
    # debug=True es útil para desarrollo, quitar para producción
    # host='0.0.0.0' permite conexiones desde otras máquinas en la red
    app.run(debug=True, host='0.0.0.0', port=5000)