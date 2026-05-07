import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
from scipy.stats import skew # Importamos 'skew' para calcular la asimetría (skewness) matemática del espectro

# Ocultamos avisos técnicos de las librerías para mantener la terminal limpia
warnings.filterwarnings('ignore')

# ==========================================
# 1. DEFINICIÓN DE RUTAS (SISTEMA DE ARCHIVOS)
# ==========================================
ruta_script = Path(__file__).parent 
carpeta_datos = ruta_script.parent / 'DATOS' 
carpeta_audios = carpeta_datos / 'Audios' 
ruta_csv_datos = carpeta_datos / "ISD v1.0 Data.csv" 

# Definimos primero la carpeta de gráficas
carpeta_graficas_audio_base = ruta_script / 'Graficas_Audios_TFG'
carpeta_graficas_audio_base.mkdir(parents=True, exist_ok=True)

ruta_csv_filtrado = carpeta_graficas_audio_base / "ISD_Cruce_Audios_Filtrado.csv"

# ==========================================
# 2. FUNCIONES DE EXTRACCIÓN DE CARACTERÍSTICAS
# ==========================================

def extraer_caracteristicas_audio(audio_path):
    """
    Extrae características del dominio de la frecuencia y del tiempo.
    Aplica ventana Hanning con 50% de solapamiento (Overlap).
    Calcula Media y Desviación Estándar de cada parámetro.
    """
    y, sr = librosa.load(audio_path, sr=None)
    
    # --- CONFIGURACIÓN DEL ENVENTANADO (50% Overlap) ---
    n_fft_val = 2048
    hop_length_val = n_fft_val // 2 # 1024 muestras de salto = 50% solapamiento

    # --- 1. Pitch (Frecuencia Fundamental o F0) ---
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), 
                                                 frame_length=n_fft_val, hop_length=hop_length_val)
    pitch_clean = f0[np.isfinite(f0)] 
    pitch_mean = np.mean(pitch_clean) if len(pitch_clean) > 0 else 0
    pitch_std = np.std(pitch_clean) if len(pitch_clean) > 0 else 0

    # --- 2. Cromagrama ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft_val, hop_length=hop_length_val)

    # --- 3. Momentos Estadísticos Espectrales ---
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft_val, hop_length=hop_length_val)
    spread = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft_val, hop_length=hop_length_val)
    S = np.abs(librosa.stft(y, n_fft=n_fft_val, hop_length=hop_length_val, window='hann'))
    skewness = skew(S, axis=0) 

    # --- 4. Flatness (Planitud Espectral) ---
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft_val, hop_length=hop_length_val)

    # --- 5. Flux (Flujo Espectral) ---
    flux = librosa.onset.onset_strength(y=y, sr=sr, n_fft=n_fft_val, hop_length=hop_length_val)

    # --- 6. Roll-off ---
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft_val, hop_length=hop_length_val, roll_percent=0.85)

    # --- 7. Inarmonicidad ---
    if pitch_mean > 0:
        inharm_array = np.abs(f0 - np.round(f0 / pitch_mean) * pitch_mean) / pitch_mean
        inharm_clean = inharm_array[np.isfinite(inharm_array)]
        inharm_mean = np.mean(inharm_clean) if len(inharm_clean) > 0 else 0
        inharm_std = np.std(inharm_clean) if len(inharm_clean) > 0 else 0
    else:
        inharm_mean, inharm_std = 0, 0

    # --- 8. RMS (Energía/Volumen) ---
    rms = librosa.feature.rms(y=y, frame_length=n_fft_val, hop_length=hop_length_val)

    # --- 9. Zero-Crossing Rate (Cruces por cero) ---
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft_val, hop_length=hop_length_val)

    # --- 10. MFCCs (1 al 20) ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=n_fft_val, hop_length=hop_length_val)
    
    # Empaquetamos todo calculando MEDIA y DESVIACIÓN ESTÁNDAR
    res = {
        'pitch_f0_hz_mean': pitch_mean, 'pitch_f0_hz_std': pitch_std,
        'inharmonicity_mean': inharm_mean, 'inharmonicity_std': inharm_std,
        'chroma_mean': np.mean(chroma), 'chroma_std': np.std(chroma),
        'spectral_centroid_mean': np.mean(centroid), 'spectral_centroid_std': np.std(centroid),
        'spectral_spread_mean': np.mean(spread), 'spectral_spread_std': np.std(spread),
        'spectral_skewness_mean': np.mean(skewness), 'spectral_skewness_std': np.std(skewness),
        'spectral_flatness_mean': np.mean(flatness), 'spectral_flatness_std': np.std(flatness),
        'spectral_flux_mean': np.mean(flux), 'spectral_flux_std': np.std(flux),
        'spectral_rolloff_mean': np.mean(rolloff), 'spectral_rolloff_std': np.std(rolloff),
        'rms_energy_mean': np.mean(rms), 'rms_energy_std': np.std(rms),
        'zero_crossing_rate_mean': np.mean(zcr), 'zero_crossing_rate_std': np.std(zcr)
    }

    # Añadimos los 20 MFCCs (Media y STD)
    for i in range(1, 21):
        res[f'mfcc_{i}_mean'] = np.mean(mfccs[i-1])
        res[f'mfcc_{i}_std'] = np.std(mfccs[i-1])

    return res

def calcular_y_guardar_caracteristicas_csv():
    """Opción 5: Extrae las matemáticas de los audios y genera el CSV final."""
    print("\n" + "="*50)
    print(" EXTRACCIÓN MASIVA DE CARACTERÍSTICAS ACÚSTICAS ")
    print("="*50)

    if not ruta_csv_filtrado.exists():
        print(f"❌ ERROR: Primero debes ejecutar la opción 4 para crear el archivo en {carpeta_graficas_audio_base.name}")
        return

    print(f"Cargando lista desde: {ruta_csv_filtrado.name}...")
    df = pd.read_csv(ruta_csv_filtrado, sep=';', decimal=',') 
    
    datos_acusticos = [] 
    print(f"Procesando {len(df)} archivos...")

    for index, row in df.iterrows():
        nombre_audio = str(row['nombre_audio_empleado'])
        archivos = list(carpeta_audios.rglob(f"{nombre_audio}.wav"))
        
        if archivos:
            audio_path = archivos[0]
            print(f"[{index+1}/{len(df)}] Analizando: {nombre_audio}...", end="\r")
            try:
                features = extraer_caracteristicas_audio(audio_path)
                fila_completa = {
                    'locationid': row.get('locationid'),
                    'groupid': row.get('groupid'),
                    'recordid': row.get('recordid'),
                    'nombre_audio': nombre_audio
                }
                fila_completa.update(features)
                datos_acusticos.append(fila_completa)
            except Exception as e:
                print(f"\n❌ Error en {nombre_audio}: {e}")
        else:
            print(f"\n⚠️ Audio no encontrado: {nombre_audio}")

    df_final = pd.DataFrame(datos_acusticos)
    ruta_final = carpeta_graficas_audio_base / "ISD_Caracteristicas_Frecuenciales.csv"
    df_final.to_csv(ruta_final, index=False, sep=';', decimal=',')
    
    print(f"\n\n✅ ¡ÉXITO! Datos guardados en: {ruta_final.name}")

# ==========================================
# 3. FUNCIONES VISUALES Y DE CRUCE DE DATOS
# ==========================================

def cruzar_audios_con_csv():
    """Opción 4: Cruza el CSV original con los archivos .wav y limpia los 'Null'."""
    print("\n" + "="*50)
    print(" CRUZANDO AUDIOS CON ENCUESTAS ")
    print("="*50)
    if not ruta_csv_datos.exists():
        print("❌ El CSV original no existe.")
        return
    
    lista_audios = list(carpeta_audios.rglob("*.wav"))
    nombres_audios_disponibles = [audio.stem.lower().strip() for audio in lista_audios]
    
    df_datos = pd.read_csv(ruta_csv_datos, low_memory=False)
    df_datos.columns = df_datos.columns.str.lower()
    
    cols_necesarias = ['locationid', 'sessionid', 'groupid', 'recordid', 'recordinglength']
    cols_presentes = [c for c in cols_necesarias if c in df_datos.columns]
    
    resultados = []
    for _, row in df_datos.iterrows():
        gid = str(row.get('groupid', '')).lower().strip()
        nombre = gid if gid in nombres_audios_disponibles else "Null"
        
        fila = row[cols_presentes].to_dict()
        fila['nombre_audio_empleado'] = nombre
        resultados.append(fila)
    
    df_res = pd.DataFrame(resultados)
    df_res = df_res[df_res['nombre_audio_empleado'] != "Null"]
    
    # Guardamos en la nueva ubicación solicitada
    df_res.to_csv(ruta_csv_filtrado, index=False, sep=';', decimal=',')
    print(f"✅ ¡Cruce finalizado! Archivo guardado en: {carpeta_graficas_audio_base.name}/{ruta_csv_filtrado.name}")

# (Resto de funciones: pedir_y_buscar_audio, generar_espectrogramas, etc.)
def pedir_y_buscar_audio():
    print("\n" + "-"*40)
    entrada_usuario = input("👉 Escribe el nombre del audio o '0' para cancelar: ").strip()
    if entrada_usuario.lower() in ['0', 'cancelar', 'salir', '']: return None, None
    nombre_audio_buscar = entrada_usuario if entrada_usuario.lower().endswith('.wav') else entrada_usuario + ".wav"
    archivos_encontrados = list(carpeta_audios.rglob(nombre_audio_buscar))
    if not archivos_encontrados:
        print(f"❌ No se encontró el archivo.")
        return None, None
    audio_path = archivos_encontrados[0]
    return audio_path, audio_path.parent.name

def obtener_carpeta_guardado_audio(tipo_grafica, ubicacion):
    carpeta_destino = carpeta_graficas_audio_base / tipo_grafica / ubicacion
    carpeta_destino.mkdir(parents=True, exist_ok=True)
    return carpeta_destino

def generar_espectrograma_lineal():
    audio_path, ubicacion = pedir_y_buscar_audio()
    if not audio_path: return
    try:
        y, sr = librosa.load(audio_path, sr=None) 
        stft = librosa.stft(y)
        espectrograma_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(espectrograma_db, sr=sr, x_axis='time', y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Espectrograma Lineal - {audio_path.name}')
        carpeta_destino = obtener_carpeta_guardado_audio('Espectrogramas_Lineales', ubicacion)
        plt.savefig(carpeta_destino / f'Lineal_{audio_path.stem}.png', bbox_inches='tight')
        plt.close()
        print(f"✅ Guardado.")
    except Exception as e: print(f"❌ Error: {e}")

def generar_espectrograma_mel():
    audio_path, ubicacion = pedir_y_buscar_audio()
    if not audio_path: return
    try:
        y, sr = librosa.load(audio_path, sr=None) 
        espectrograma = librosa.feature.melspectrogram(y=y, sr=sr)
        espectrograma_db = librosa.power_to_db(espectrograma, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(espectrograma_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Espectrograma de Mel - {audio_path.name}')
        carpeta_destino = obtener_carpeta_guardado_audio('Espectrogramas_Mel', ubicacion)
        plt.savefig(carpeta_destino / f'Mel_{audio_path.stem}.png', bbox_inches='tight')
        plt.close()
        print(f"✅ Guardado.")
    except Exception as e: print(f"❌ Error: {e}")

def generar_mfcc():
    audio_path, ubicacion = pedir_y_buscar_audio()
    if not audio_path: return
    try:
        y, sr = librosa.load(audio_path, sr=None) 
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.colorbar()
        plt.title(f'MFCC - {audio_path.name}')
        carpeta_destino = obtener_carpeta_guardado_audio('MFCC', ubicacion)
        plt.savefig(carpeta_destino / f'MFCC_{audio_path.stem}.png', bbox_inches='tight')
        plt.close()
        print(f"✅ Guardado.")
    except Exception as e: print(f"❌ Error: {e}")

# ==========================================
# 4. PUNTO DE ENTRADA (MENÚ INTERACTIVO)
# ==========================================
if __name__ == "__main__":
    while True:
        print("\n" + "*"*40)
        print(" MENÚ DE ANÁLISIS DE AUDIO TFG ")
        print("*"*40)
        print("1. Espectrograma Lineal (Hz)")
        print("2. Espectrograma de Mel")
        print("3. Gráfico MFCC (Visual)")
        print("4. Mapear audios con CSV original (PASO PREVIO AL 6)")
        print("5. EXTRAER CARACTERÍSTICAS A CSV (F0, Inarmonicidad, MFCCs, etc.)")
        print("0. Salir")
        
        opcion = input("\nElige una opción (0-5): ").strip()
        if opcion == '1': generar_espectrograma_lineal()
        elif opcion == '2': generar_espectrograma_mel()
        elif opcion == '3': generar_mfcc()
        elif opcion == '4': cruzar_audios_con_csv()
        elif opcion == '5': calcular_y_guardar_caracteristicas_csv()
        elif opcion == '0': break