from flask import Flask, request, jsonify, send_file
from flask_cors import CORS # Importar Flask-CORS
import os
from glob import iglob
from pathlib import Path
import asv

# App que implementa el servidor de Flask
app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas automáticamente

# Carpeta para almacenar registros de voz
DATASET_FOLDER = "audios"
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Diccionario donde almacenaremos las huellas vocales de los usuarios ya registrados ()
reg_user_vocal_fingerprint = dict()

# Calculamos las huellas vocales de todos los audios en el directorio audios/
for wav_file in iglob(os.path.join('audios', '*.wav')):
    user_id = Path(wav_file).stem   # Nombre del ficher sin extensión
    reg_user_vocal_fingerprint[user_id] = asv.compute_vocal_fingerprint(wav_file)


@app.route("/")
def home_page():
    return send_file('static/index.html')

# Registro de voz (almacena la muestra con la ID del usuario)
@app.route("/register", methods=["POST"])
def register():
    user_id = request.form.get("user_id")
    if not user_id:
        return jsonify({"error": "Falta el ID del usuario"}), 400
    
    if user_id in reg_user_vocal_fingerprint:
        return jsonify({"error": "Usuario ya registrado"}), 400
    
    file = request.files.get("audio")
    if not file:
        return jsonify({"error": "No se proporcionó un archivo de audio"}), 400
    
    # Almacenamos el audio del nuevo usuario en la carpeta audios y calculamos su huella vocal
    print(f'Registrando a: {user_id}')
    audio_path = os.path.join(DATASET_FOLDER, f"{user_id}.wav")
    file.save(audio_path)
    reg_user_vocal_fingerprint[user_id] = asv.compute_vocal_fingerprint(audio_path)
    
    return jsonify({"message": "Registro completado!!"})


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
        return jsonify({"error": "El usuario no está registrado"}), 400
    
    print(f'Verificando a: {user_id}')
    # Calcular la huella vocal de la nueva muestra de audio
    test_audio_path = "temp.wav"
    file.save(test_audio_path)
    test_fingerprint = asv.compute_vocal_fingerprint(test_audio_path)
    
    # Comparamos ambas huellas vocales
    verified, distance = asv.compare_vocal_fingerprints(reg_user_vocal_fingerprint[user_id], test_fingerprint)  
    print(f'Verified: {verified} - Distance: {distance}')

    return jsonify({"verified": verified, "distance": distance})


if __name__ == "__main__":
    app.run(debug=True)
