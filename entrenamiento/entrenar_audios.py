import os
import sys
import numpy as np
import joblib
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
sys.path.append(os.path.abspath(".."))

from asv import compute_vocal_fingerprint

BASE_DIR = "es"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

user_dirs = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]

for user_id in user_dirs:
    folder = os.path.join(BASE_DIR, user_id)
    all_files = [f for f in os.listdir(folder) if f.endswith(".wav")]

    selected = sorted(all_files)[:20]  # Limitar a 20 archivos

    print(f"\n🔧 Entrenando modelo para: {user_id} con {len(selected)} audios...")

    fingerprints = []
    for fname in selected:
        path = os.path.join(folder, fname)
        print(f"🎙️ Procesando: {fname}")
        fp = compute_vocal_fingerprint(path)
        if fp is not None:
            fingerprints.append(fp)
        else:
            print(f"⚠️ Audio inválido: {fname}")

    if len(fingerprints) < 10:
        print(f"❌ No suficientes huellas para {user_id}, requiere al menos 10")
        continue

    X = np.stack(fingerprints)

    # Normalizar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entrenar SVM con parámetros ajustados
    model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
    model.fit(X_scaled)

    # Guardar modelo y scaler
    joblib.dump(model, os.path.join(MODEL_DIR, f"{user_id}_svm.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{user_id}_scaler.pkl"))

    print(f"✅ Modelo y scaler guardados para: {user_id}")
