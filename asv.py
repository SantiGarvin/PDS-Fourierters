import librosa
from scipy.spatial.distance import euclidean, cosine # Importar cosine
import numpy as np
import logging # Añadir logging

# --- Constantes de Configuración ---
DEFAULT_SR = 16000      # Frecuencia de muestreo por defecto
N_MFCC = 13             # Número de coeficientes MFCC a extraer
HOP_LENGTH = 512        # Salto entre ventanas para MFCC (librosa default)
N_FFT = 2048            # Tamaño de la FFT para MFCC (librosa default)

# Umbral para la distancia coseno. Valores más *pequeños* indican mayor similitud.
# (Cosine distance = 1 - cosine similarity). Un valor típico para match puede ser < 0.4 o 0.5
# ¡Este valor probablemente necesite ser ajustado empíricamente!
COSINE_THRESHOLD = 0.4

def compute_vocal_fingerprint(audio_path, sr=DEFAULT_SR, n_mfcc=N_MFCC):
    """
    Extrae la huella vocal de un audio usando la media y desviación estándar de los MFCCs.

    Args:
        audio_path (str): Ruta del archivo de audio.
        sr (int, opcional): Frecuencia de muestreo. Por defecto DEFAULT_SR.
        n_mfcc (int, opcional): Número de coeficientes MFCC. Por defecto N_MFCC.

    Returns:
        np.ndarray: Vector de características concatenando la media y la desviación estándar
                    de cada coeficiente MFCC, o None si ocurre un error.
    """
    try:
        # Cargar audio con la frecuencia de muestreo especificada
        y, current_sr = librosa.load(audio_path, sr=sr)

        # Verificar si el audio es muy corto o está vacío
        if len(y) < N_FFT: # Si es más corto que una ventana FFT, MFCC fallará o será inválido
             logging.warning(f"Audio en {audio_path} es demasiado corto para extraer características.")
             return None

        # Extraer los coeficientes MFCC
        # Usar n_fft y hop_length puede dar más control, pero los defaults suelen funcionar bien
        mfccs = librosa.feature.mfcc(y=y, sr=current_sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH)

        # Verificar si se obtuvieron MFCCs (podría fallar con silencio puro)
        if mfccs.shape[1] == 0:
            logging.warning(f"No se pudieron extraer MFCCs de {audio_path} (posiblemente silencio).")
            return None

        # Calcular la media y desviación estándar de cada coeficiente
        mean_mfccs = np.mean(mfccs, axis=1)
        std_mfccs = np.std(mfccs, axis=1)

        # Concatenar media y std para formar la huella vocal
        fingerprint = np.concatenate((mean_mfccs, std_mfccs))

        return fingerprint

    except Exception as e:
        logging.error(f"Error al procesar {audio_path}: {e}")
        # Podrías querer manejar tipos específicos de excepciones de librosa aquí
        return None # Devolver None para indicar fallo

def compare_vocal_fingerprints(fp1, fp2, threshold=COSINE_THRESHOLD):
    """
    Compara dos huellas vocales utilizando la distancia Coseno.

    Args:
        fp1 (np.ndarray): Primera huella vocal (media + std MFCCs).
        fp2 (np.ndarray): Segunda huella vocal (media + std MFCCs).
        threshold (float, opcional): Umbral de decisión para la distancia Coseno.
                                     Valores más *bajos* indican mayor similitud.
                                     Por defecto es COSINE_THRESHOLD.

    Returns:
        tuple:
            - bool: True si la distancia es menor que el umbral (misma persona), False en caso contrario.
            - float: Valor de la distancia Coseno calculada.
    Nota:
        - La distancia Coseno varía entre 0 (vectores idénticos) y 2 (vectores opuestos).
        - Un valor de 1 indica ortogonalidad.
        - Se espera que huellas de la misma persona tengan una distancia coseno baja (< threshold).
    """
    if fp1 is None or fp2 is None or fp1.shape != fp2.shape:
         logging.error("Comparación fallida: Huellas inválidas o con formas diferentes.")
         # Devolver False y una distancia 'infinita' o muy grande para indicar fallo/no coincidencia
         return False, np.inf # O 2.0, que es el máximo teórico de la distancia coseno

    # Asegurarse de que los vectores no sean nulos (lo que daría NaN en cosine)
    if np.all(fp1 == 0) or np.all(fp2 == 0):
        logging.warning("Comparación con vector cero detectada.")
        # Si ambos son cero, son 'iguales' (distancia 0), si uno es cero y otro no, son muy diferentes (distancia 1 o más)
        # La función cosine maneja esto, pero podemos ser explícitos si queremos.
        # Por simplicidad, dejamos que 'cosine' lo maneje (puede dar NaN si la norma es 0).
        # O podemos devolver directamente:
        # return np.all(fp1 == 0) and np.all(fp2 == 0), 1.0 if (np.all(fp1==0) ^ np.all(fp2==0)) else 0.0

    distance = cosine(fp1, fp2)

    # Manejar posible NaN si alguna norma es cero (aunque improbable con MFCCs reales)
    if np.isnan(distance):
        logging.warning("Distancia Coseno resultó en NaN.")
        return False, np.inf # Tratar NaN como no coincidencia

    verified = distance < threshold
    return verified, float(distance) # Asegurar que la distancia es un float estándar