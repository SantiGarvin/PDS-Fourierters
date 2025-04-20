from flask import Flask, request, jsonify, send_file
from flask_cors import CORS # Importar Flask-CORS
import os
from glob import iglob
from pathlib import Path
import asv
import json
import logging
import numpy as np
import tempfile
import shutil

# Configurar logging básico
logging.basicConfig(level=logging.INFO)

# App que implementa el servidor de Flask
app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas automáticamente

# Carpeta para almacenar registros de voz
DATASET_FOLDER = "audios"
FINGERPRINT_DB_FILE = "fingerprints.db.json" # Archivo para persistencia
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Diccionario donde almacenaremos las huellas vocales de los usuarios ya registrados ()
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
         logging.error(f"Error de tipo al serializar huellas a JSON: {e}")


# --- Carga inicial de huellas ---
load_fingerprints()

# # Calculamos las huellas vocales de todos los audios en el directorio audios/
# for wav_file in iglob(os.path.join('audios', '*.wav')):
#     user_id = Path(wav_file).stem   # Nombre del ficher sin extensión
#     reg_user_vocal_fingerprint[user_id] = asv.compute_vocal_fingerprint(wav_file)


@app.route("/")
def home_page():
    return send_file('static/index.html')

# Registro de voz
@app.route("/register", methods=["POST"])
def register():
    user_id = request.form.get("user_id")
    if not user_id:
        return jsonify({"error": "Falta el ID del usuario"}), 400

    # Validar formato de user_id (ej: no vacío, sin caracteres extraños)
    if not user_id.strip():
         return jsonify({"error": "ID de usuario inválido"}), 400

    if user_id in reg_user_vocal_fingerprint:
        return jsonify({"error": f"Usuario '{user_id}' ya registrado"}), 400 # 409 Conflict sería más semántico

    file = request.files.get("audio")
    if not file:
        return jsonify({"error": "No se proporcionó un archivo de audio"}), 400

    # Guardamos el audio del nuevo usuario en la carpeta audios (opcional, pero útil para depurar/reentrenar)
    audio_path = os.path.join(DATASET_FOLDER, f"{user_id}.wav")
    try:
        file.save(audio_path)
        logging.info(f"Audio de registro guardado en: {audio_path}")
    except Exception as e:
        logging.error(f"Error al guardar el archivo de audio {audio_path}: {e}")
        return jsonify({"error": "Error interno al guardar el audio"}), 500

    # Calculamos su huella vocal
    try:
        new_fingerprint = asv.compute_vocal_fingerprint(audio_path)
        reg_user_vocal_fingerprint[user_id] = new_fingerprint
        logging.info(f'Registrando a: {user_id}')

        # Guardar el diccionario actualizado en el archivo JSON
        save_fingerprints()

        return jsonify({"message": f"Usuario '{user_id}' registrado correctamente."})

    except Exception as e:
        # Si falla el cálculo de la huella, no deberíamos dejar el usuario registrado
        # ni el archivo de audio
        logging.error(f"Error al calcular la huella vocal para {user_id}: {e}")
        # Opcional: eliminar el archivo de audio si falla el procesamiento
        if os.path.exists(audio_path):
             try:
                 os.remove(audio_path)
                 logging.warning(f"Archivo de audio {audio_path} eliminado debido a error de procesamiento.")
             except OSError as rm_err:
                 logging.error(f"Error al intentar eliminar {audio_path}: {rm_err}")
        return jsonify({"error": "Error interno al procesar el audio"}), 500


# Verificación de locutor
@app.route("/verify", methods=["POST"])
def verify():
    user_id = request.form.get("user_id")
    if not user_id:
        return jsonify({"error": "Falta el ID del usuario"}), 400

    file = request.files.get("audio")
    if not file:
        return jsonify({"error": "No se proporcionó un archivo de audio"}), 400

    if user_id not in reg_user_vocal_fingerprint:
        return jsonify({"error": f"El usuario '{user_id}' no está registrado"}), 404

    logging.info(f'Verificando a: {user_id}')

    # Guardar temporalmente el audio de verificación
    # MEJORA PENDIENTE: Usar un archivo temporal seguro
    test_audio_path = "temp_verify.wav" # Cambiado para evitar colisión con registro si falla
    try:
        # Crear un archivo temporal con extensión .wav
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            test_audio_path = tmp_file.name
            file.save(test_audio_path) # Guardar el contenido del archivo subido en el temporal

        logging.info(f"Audio de verificación guardado temporalmente en: {test_audio_path}")

        # Calcular la huella vocal de la nueva muestra de audio
        test_fingerprint = asv.compute_vocal_fingerprint(test_audio_path)

        # Comparamos ambas huellas vocales
        registered_fingerprint = reg_user_vocal_fingerprint[user_id]
        verified, distance = asv.compare_vocal_fingerprints(registered_fingerprint, test_fingerprint)

        # Asegurarse de que 'verified' sea un booleano Python nativo para JSON
        verified_bool = bool(verified)
        # Asegurarse de que 'distance' sea un float Python nativo para JSON
        distance_float = float(distance)

        logging.info(f'Verified: {verified_bool} - Distance: {distance_float}')

        # Limpiar el archivo temporal después de usarlo
        if os.path.exists(test_audio_path):
            os.remove(test_audio_path)
            logging.info(f"Archivo temporal {test_audio_path} eliminado.")
            
        return jsonify({"verified": verified_bool, "distance": distance_float})

    except FileNotFoundError:
         logging.error(f"Error: El archivo temporal {test_audio_path} no se encontró después de guardarlo.")
         return jsonify({"error": "Error interno del servidor al manejar el archivo"}), 500
    except Exception as e:
        logging.error(f"Error durante la verificación para {user_id}: {e}")
        # Eliminar el archivo de audio temporal si falla el procesamiento
        if os.path.exists(test_audio_path):
            try:
                os.remove(test_audio_path)
                logging.warning(f"Archivo temporal {test_audio_path} eliminado debido a error de procesamiento.")
            except OSError as rm_err:
                logging.error(f"Error al intentar eliminar {test_audio_path}: {rm_err}")

        # Asegurarse de limpiar el archivo temporal también en caso de error
        if 'test_audio_path' in locals() and os.path.exists(test_audio_path):
            try:
                os.remove(test_audio_path)
                logging.warning(f"Archivo temporal {test_audio_path} eliminado tras error.")
            except OSError as rm_err:
                logging.error(f"Error al intentar eliminar {test_audio_path} tras error: {rm_err}")
        return jsonify({"error": "Error interno al procesar la verificación"}), 500

if __name__ == "__main__":
    # debug=True es útil para desarrollo, pero recuerda quitarlo para producción
    app.run(debug=True)