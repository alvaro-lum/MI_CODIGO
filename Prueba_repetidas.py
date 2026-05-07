import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ==========================================
ruta_script = Path(__file__).parent
carpeta_datos = ruta_script.parent / 'DATOS'

ruta_csv = carpeta_datos / "ISD v1.0 Data.csv"

repetidas = ruta_script / 'Repetidas'
repetidas.mkdir(parents=True, exist_ok=True)

print("--- Comprobación de archivos ---")
print(f"Buscando en: {carpeta_datos}")
print(f"¿Encuentra el CSV?: {ruta_csv.exists()}")
print("-" * 30)

print("\nCargando y depurando datos...")
try:
    df_datos = pd.read_csv(ruta_csv, low_memory=False)
    df_datos.columns = df_datos.columns.str.lower()
    
    filas_originales = df_datos.shape[0]
    print(f"📥 CSV original cargado. Contiene {filas_originales} filas en total.")

    # ---------------------------------------------------------
    # PASO 1: LA MISMA DEPURACIÓN DE NULOS QUE EN ANALISIS.PY
    # ---------------------------------------------------------
    columnas_opcionales = ['groupid','start_time','end_time','latitude','longitude','language',
                           'survey_version','ssi01','ssi02','ssi03','ssi04','sss01','sss02','sss03','sss04','sss05','who01','who02','who03','who04','who05',
                           'who_sum','age00','edu00','eth00','occ00','eth00_other', 'occ00_other','misc03', 'misc03_other', 'misc01','use00', 'uni00',
                             'recordinglength']
    
    columnas_opcionales = [col.lower() for col in columnas_opcionales]
    columnas_utiles = [col for col in df_datos.columns if col not in columnas_opcionales]

    # Eliminamos las filas incompletas
    df_datos = df_datos.dropna(subset=columnas_utiles, how='any')
    
    filas_sin_nulos = df_datos.shape[0]
    print(f"🧹 DEPURACIÓN: Se han eliminado {filas_originales - filas_sin_nulos} filas por datos incompletos (nulos).")

    # ---------------------------------------------------------
    # PASO 2: EXTRAER LAS FILAS REPETIDAS (EN VEZ DE BORRARLAS)
    # ---------------------------------------------------------
    if 'recordid' in df_datos.columns and 'locationid' in df_datos.columns:
        
        # El parámetro keep=False atrapa TODAS las encuestas que compartan un mismo recordid en la misma locationid
        mascara_duplicados = df_datos.duplicated(subset=['recordid', 'locationid'], keep=False)
        
        # Filtramos para quedarnos EXCLUSIVAMENTE con los repetidos
        df_repetidos = df_datos[mascara_duplicados].copy()
        
        # Los ordenamos para que el original y su copia aparezcan juntos en el Excel (fáciles de comparar visualmente)
        df_repetidos = df_repetidos.sort_values(by=['locationid', 'recordid'])

        filas_repetidas = df_repetidos.shape[0]
        
        if filas_repetidas > 0:
            print(f"⚠️ Se han encontrado {filas_repetidas} filas involucradas en repeticiones (tras limpiar nulos).")
            
            # Guardamos el CSV
            ruta_exportacion = repetidas / "ISD_Filas_Repetidas_TFG.csv"
            df_repetidos.to_csv(ruta_exportacion, index=False, sep=';', decimal=',')
            print(f"💾 ¡ÉXITO! Se ha guardado el CSV con las filas repetidas en: {ruta_exportacion.name}")
        else:
            print("✅ ¡Perfecto! No hay encuestas repetidas en tu base de datos tras limpiarla.")
            
    else:
        print("❌ Error: Faltan las columnas 'recordid' o 'locationid' para buscar duplicados.")

except Exception as e:
    print(f"❌ Error al cargar o depurar el CSV: {e}")