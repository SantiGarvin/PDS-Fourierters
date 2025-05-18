import librosa
import librosa.effects
from scipy.spatial.distance import cosine
import numpy as np
import logging

import soundfile as sf

from scipy.signal import butter, lfilter




# --- Constantes de Configuración ---
DEFAULT_SR = 16000      # Frecuencia de muestreo por defecto
N_MFCC = 13             # Número de coeficientes MFCC base a extraer
HOP_LENGTH = 512        # Salto entre ventanas para MFCC
N_FFT = 2048            # Tamaño de la FFT para MFCC

# Parámetros para el recorte de silencio simple basado en energía
SILENCE_THRESHOLD_DB = 40 # Umbral en dB por debajo del pico máximo para considerar silencio. AJUSTAR si es necesario.

# Umbral para la distancia coseno. Valores más *pequeños* indican mayor similitud.
# AHORA LA HUELLA ES MÁS GRANDE (MFCC+Delta+Delta2, Mean+Std).
# ¡¡¡ESTE UMBRAL NECESITARÁ SER REAJUSTADO SIGNIFICATIVAMENTE!!!
# Empezar probando valores entre 0.4 y 0.7 quizás.
COSINE_THRESHOLD = 0.5 # Valor inicial, necesita ajuste empírico.


def bandpass_filter(y, sr, low=300, high=3400, order=4):
    """
    Aplica un filtro Butterworth pasa‑banda [low, high] Hz.
    """
    nyq = sr / 2
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return lfilter(b, a, y)

# Reducción de ruido espectral (Spectral Subtraction) 
def reduce_noise(y, sr, n_fft=N_FFT, hop_length=HOP_LENGTH, noise_duration=0.5, prop_decrease=1.0):
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    # Número de frames para estimar ruido
    n_noise = int(noise_duration * sr / hop_length)
    noise_mag = np.mean(mag[:, :n_noise], axis=1, keepdims=True)
    # Subtracción y clamp
    mag_clean = np.maximum(mag - prop_decrease * noise_mag, 0.0)
    # Reconstrucción
    stft_clean = mag_clean * np.exp(1j*phase)
    return librosa.istft(stft_clean, hop_length=hop_length)


def normalize_frames(m,epsilon=1e-8):
    # Normalización por enunciado (CMVN simple): restar la media y dividir por std
    # m: matriz de características (ej: MFCCs) de tamaño (n_features, n_frames)
    return (m - np.mean(m, axis=1, keepdims=True)) / (np.std(m, axis=1, keepdims=True) + epsilon)

def compute_vocal_fingerprint(audio_path, sr=DEFAULT_SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, silence_thresh=SILENCE_THRESHOLD_DB):
    """
    Extrae la huella vocal de un audio usando:
    1. Recorte de silencios iniciales/finales.
    2. MFCCs, Delta MFCCs, Delta-Delta MFCCs.
    3. Normalización Cepstral por Enunciado (CMVN simple).
    4. Media y Desviación Estándar de las características normalizadas.

    Args:
        audio_path (str): Ruta del archivo de audio.
        sr (int): Frecuencia de muestreo deseada.
        n_mfcc (int): Número de coeficientes MFCC base.
        n_fft (int): Tamaño de la ventana FFT.
        hop_length (int): Salto de la ventana.
        silence_thresh (float): Umbral en dB para el recorte de silencio.

    Returns:
        np.ndarray: Vector de características (huella vocal) concatenado,
                    o None si ocurre un error o el audio es demasiado corto/silencioso.
    """
    try:
        # 1. Cargar audio
        y, file_sr = sf.read(audio_path)
        if y.ndim > 1:
            y = y.mean(axis=1)  # convertir a mono
        if file_sr != sr:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
        current_sr = sr
        
        # 2. Reducción de ruido
        y = reduce_noise(y, sr,
                         n_fft=n_fft,
                         hop_length=hop_length,
                         noise_duration=0.5,
                         prop_decrease=1.0)
        
        # 3. Aplicamos el filtro de banda
        y = bandpass_filter(y, current_sr)

        # 4. Recorte de silencios iniciales/finales
        y_trimmed, index = librosa.effects.trim(y, top_db=silence_thresh, frame_length=n_fft, hop_length=hop_length)
        logging.info(f"Audio original: {len(y)/current_sr:.2f}s. Recortado: {len(y_trimmed)/current_sr:.2f}s.")

        # Verificar si queda audio después del recorte
        if len(y_trimmed) < n_fft: # Necesita al menos una ventana FFT
             logging.warning(f"Audio en {audio_path} es demasiado corto o silencioso después del recorte.")
             return None

        # 5. Extraer MFCCs base
        mfccs = librosa.feature.mfcc(y=y_trimmed, sr=current_sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        # 6. Calcular Deltas y Delta-Deltas
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        # Verificar si se obtuvieron suficientes frames
        if mfccs.shape[1] == 0 or delta_mfccs.shape[1] == 0 or delta2_mfccs.shape[1] == 0:
            logging.warning(f"No se pudieron extraer suficientes frames de características de {audio_path}.")
            return None

        # 7. Aplicar Normalización Cepstral (CMVN por enunciado simple)
        mfccs_norm = normalize_frames(mfccs)
        delta_mfccs_norm = normalize_frames(delta_mfccs)
        delta2_mfccs_norm = normalize_frames(delta2_mfccs)

        # 8. Calcular Media y Desviación Estándar de cada tipo de característica normalizada
        mean_mfccs = np.mean(mfccs_norm, axis=1)
        std_mfccs = np.std(mfccs_norm, axis=1)
        mean_delta = np.mean(delta_mfccs_norm, axis=1)
        std_delta = np.std(delta_mfccs_norm, axis=1)
        mean_delta2 = np.mean(delta2_mfccs_norm, axis=1)
        std_delta2 = np.std(delta2_mfccs_norm, axis=1)

        # 9. Concatenar todo en una única huella vocal
        # Tamaño: (n_mfcc * 2) + (n_mfcc * 2) + (n_mfcc * 2) = n_mfcc * 6
        fingerprint = np.concatenate((
            mean_mfccs, std_mfccs,
            mean_delta, std_delta,
            mean_delta2, std_delta2
        ))

        # Verificar si hay NaNs (podría ocurrir si std es cero en normalize_frames)
        if np.isnan(fingerprint).any():
            logging.error(f"NaN detectado en la huella final de {audio_path}. Posiblemente audio constante o muy corto.")
            return None

        return fingerprint

    except Exception as e:
        logging.error(f"Error al procesar {audio_path}: {e}", exc_info=True) # Añadir stack trace al log
        return None # Devolver None para indicar fallo


# La función de comparación no necesita cambios, pero el UMBRAL sí.
def compare_vocal_fingerprints(fp1, fp2, threshold=COSINE_THRESHOLD):
    """
    Compara dos huellas vocales (MFCC+Delta+Delta2, Mean+Std) usando la distancia Coseno.

    Args:
        fp1 (np.ndarray): Primera huella vocal.
        fp2 (np.ndarray): Segunda huella vocal.
        threshold (float, opcional): Umbral de decisión para la distancia Coseno.
                                     ¡¡NECESITA AJUSTE EMPÍRICO!!

    Returns:
        tuple: (bool: verified, float: distance)
    """
    if fp1 is None or fp2 is None:
        logging.error("Comparación fallida: Al menos una huella es None.")
        return False, np.inf
    if fp1.shape != fp2.shape:
         logging.error(f"Comparación fallida: Huellas con formas diferentes. {fp1.shape} vs {fp2.shape}")
         return False, np.inf

    # Evitar división por cero en cosine si alguna huella es toda ceros (improbable pero posible)
    if np.all(fp1 == 0) or np.all(fp2 == 0):
        logging.warning("Comparación con vector cero detectada.")
        # Si ambos son cero, son "idénticos" (distancia 0). Si uno es cero y otro no, son máximamente diferentes (distancia 1 en similitud coseno, o >1 en distancia).
        # La función 'cosine' devuelve 1 si uno es cero y el otro no.
        distance = cosine(fp1, fp2) # Puede ser 0.0, 1.0, o NaN si ambos son 0
        if np.isnan(distance): distance = 0.0 # Si ambos son 0, consideramos distancia 0
    else:
        distance = cosine(fp1, fp2)

    # Manejar posible NaN residual (aunque menos probable ahora)
    if np.isnan(distance):
        logging.warning("Distancia Coseno resultó en NaN.")
        return False, np.inf

    verified = distance < threshold
    logging.debug(f"Distancia Coseno calculada: {distance:.4f}, Umbral: {threshold}, Verificado: {verified}")
    return verified, float(distance)