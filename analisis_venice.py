import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import soundscapy as sspy
import warnings
warnings.filterwarnings('ignore')

sns.set_context("notebook")
# ==========================================
# 1. CONFIGURACIÓN, CARGA Y DEPURACIÓN DE DATOS
# ==========================================
ruta_script = Path(__file__).parent
carpeta_datos = ruta_script.parent / 'DATOS'

ruta_csv = carpeta_datos / "ISD v1.0 Data.csv"
ruta_excel = carpeta_datos / "ISD v1.0 Metadata.xlsx"

# Definimos la NUEVA carpeta principal de gráficas de forma GLOBAL para Venecia
carpeta_graficas_base = ruta_script / 'Graficas_Venecia_TFG'
carpeta_graficas_base.mkdir(parents=True, exist_ok=True)

print("--- Comprobación de archivos ---")
print(f"Buscando en: {carpeta_datos}")
print(f"¿Encuentra el CSV?: {ruta_csv.exists()}")
print(f"¿Encuentra el Excel?: {ruta_excel.exists()}")
print("-" * 30)

print("\nCargando y depurando datos...")
try:
    df_datos = pd.read_csv(ruta_csv, low_memory=False)
    # 1. Estandarizamos todas las columnas a minúsculas
    df_datos.columns = df_datos.columns.str.lower()
    
    filas_originales = df_datos.shape[0]
    print(f"📥 CSV original cargado. Contiene {filas_originales} filas en total.")

    # 2. Definimos las columnas que NO nos importan si están vacías
    columnas_opcionales = ['groupid','start_time','end_time','latitude','longitude','language',
                           'survey_version','ssi01','ssi02','ssi03','ssi04','sss01','sss02','sss03','sss04','sss05','who01','who02','who03','who04','who05',
                           'who_sum','age00','edu00','eth00','occ00','eth00_other', 'occ00_other','misc03', 'misc03_other', 'misc01','use00', 'uni00',
                             'recordinglength']
    
    columnas_opcionales = [col.lower() for col in columnas_opcionales]
    
    # 3. Nos quedamos solo con las columnas "útiles" (las que DEBEN tener datos)
    columnas_utiles = [col for col in df_datos.columns if col not in columnas_opcionales]

    # 4. Eliminamos filas que tengan valores nulos (NaN) en las columnas útiles
    df_datos = df_datos.dropna(subset=columnas_utiles, how='any')
    
    # 5. Eliminamos encuestas duplicadas
    if 'recordid' in df_datos.columns and 'locationid' in df_datos.columns:
        df_datos = df_datos.drop_duplicates(subset=['recordid', 'locationid', 'groupid'])
    else:
        df_datos = df_datos.drop_duplicates()

    # 6. FILTRO GEOGRÁFICO: Solo localizaciones de Venecia
    localizaciones_venecia = [
        'MonumentoGaribaldi', 'SanMarco'
    ]
    
    # Nos aseguramos de ignorar mayúsculas/minúsculas al filtrar
    localizaciones_venecia_lower = [loc.lower() for loc in localizaciones_venecia]
    
    if 'locationid' in df_datos.columns:
        # Filtramos quedándonos solo con las filas cuyo locationid esté en la lista de Venecia
        df_datos = df_datos[df_datos['locationid'].str.lower().isin(localizaciones_venecia_lower)]
    else:
        print("⚠️ Advertencia: No se encontró la columna 'locationid'. No se pudo aplicar el filtro de Venecia.")

    # 7. Resumen de la depuración
    filas_limpias = df_datos.shape[0]
    filas_eliminadas = filas_originales - filas_limpias
    
    print(f"🧹 DEPURACIÓN ESTRICTA: Se han eliminado {filas_eliminadas} filas (incompletas, duplicadas o fuera de Venecia).")
    print(f"✅ Datos listos para el análisis: {filas_limpias} participantes únicos en plazas de Venecia.")
    
    # ---------------------------------------------------------
    # CÁLCULO GLOBAL DE COORDENADAS ISO AL INICIO
    # ---------------------------------------------------------
    print("\nCalculando coordenadas ISO globales (Soundscapy)...")
    try: 
        mapa_paqs = {
            'pleasant': 'PAQ1', 'vibrant': 'PAQ2', 'eventful': 'PAQ3', 'chaotic': 'PAQ4',
            'annoying': 'PAQ5', 'monotonous': 'PAQ6', 'uneventful': 'PAQ7', 'calm': 'PAQ8'
        }
        df_datos = df_datos.rename(columns=mapa_paqs)
        df_datos = sspy.surveys.add_iso_coords(df_datos, overwrite=True)
        mapa_inverso = {v: k for k, v in mapa_paqs.items()}
        df_datos = df_datos.rename(columns=mapa_inverso)
        print("✅ Coordenadas ISO (ISOPleasant, ISOEventful) añadidas al dataset global.")
    except Exception as e:
        print(f"⚠️ Error al calcular las coordenadas ISO globales: {e}")
    # ---------------------------------------------------------

    ruta_exportacion = carpeta_graficas_base / "ISD_Datos_Venecia_Limpios_TFG.csv"
    df_datos.to_csv(ruta_exportacion, index=False, sep=';', decimal=',')
    print(f"💾 ¡ÉXITO! Se ha guardado una copia limpia de los datos (con ISO) en: {ruta_exportacion.name}")

except Exception as e:
    print(f"❌ Error al cargar o depurar el CSV: {e}")

try:
    df_metadatos = pd.read_excel(ruta_excel)
    print(f"✅ Metadatos (Excel) cargados con éxito.")
except Exception as e:
    print(f"❌ Error al cargar el Excel: {e}")


# ==========================================
# FUNCIONES AYUDANTES
# ==========================================

def obtener_carpeta_guardado(localizacion="General"):
    """Crea y devuelve la ruta de una subcarpeta dentro de Graficas_Venecia_TFG."""
    carpeta_destino = carpeta_graficas_base / localizacion
    carpeta_destino.mkdir(parents=True, exist_ok=True)
    return carpeta_destino

def dar_formato_ejes_iso(ax, titulo):
    """Función ayudante para pintar la cuadrícula ISO en cualquier gráfica."""
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    ax.plot([-1, 1], [-1, 1], color='gray', linestyle='--', alpha=0.7)
    ax.plot([-1, 1], [1, -1], color='gray', linestyle='--', alpha=0.7)
    ax.text(0.5, 0.5, '(vibrant)', style='italic', ha='center', va='center', fontsize=11, color='gray', alpha=0.8)
    ax.text(-0.5, 0.5, '(chaotic)', style='italic', ha='center', va='center', fontsize=11, color='gray', alpha=0.8)
    ax.text(-0.5, -0.5, '(monotonous)', style='italic', ha='center', va='center', fontsize=11, color='gray', alpha=0.8)
    ax.text(0.5, -0.5, '(calm)', style='italic', ha='center', va='center', fontsize=11, color='gray', alpha=0.8)
    ax.set_xlabel('$P_{ISO}$', fontsize=12)
    ax.set_ylabel('$E_{ISO}$', fontsize=12)
    ax.set_title(titulo, fontsize=14, pad=15)


# ==========================================
# DEFINICIÓN DE FUNCIONES DE ANÁLISIS
# ==========================================

def resumir_y_exportar_ISO():
    global df_datos 
    print("\n" + "="*50)
    print(" RESUMEN Y EXPORTACIÓN DE PARÁMETROS ISO ")
    print("="*50)

    cols = df_datos.columns.tolist()
    columnas_deseadas = ['locationid', 'pleasant', 'eventful', 'ISOPleasant', 'ISOEventful', 'ISOPleasant_error', 'ISOEventful_error']
    columnas_finales_exactas = [col_real for col_buscada in columnas_deseadas for col_real in cols if col_buscada.lower() == col_real.lower()]

    if not columnas_finales_exactas:
        print("⚠️ No se encuentran las columnas ISO para exportar.")
        return

    df_exportar = df_datos[columnas_finales_exactas]
    ruta_exportacion = carpeta_graficas_base / "ISD_Parametros_ISO_Venecia_TFG.csv"
    df_exportar.to_csv(ruta_exportacion, index=False, sep=';', decimal=',')
    print(f"💾 CSV exclusivo con parámetros ISO guardado en: {ruta_exportacion.name}\n")

    print("-" * 40)
    print(" RESUMEN ACÚSTICO POR LOCALIZACIÓN ")
    print("-" * 40)
    
    col_loc = next((c for c in cols if c.lower() == 'locationid'), None)
    col_iso_p = next((c for c in cols if c.lower() == 'isopleasant'), None)
    col_iso_e = next((c for c in cols if c.lower() == 'isoeventful'), None)
    
    if col_loc and col_iso_p and col_iso_e:
        localizaciones = df_datos[col_loc].dropna().unique()
        for loc in localizaciones:
            loc_data = df_datos[df_datos[col_loc] == loc]
            print(f"Data for {loc}:")
            print(f"Number of records: {len(loc_data)}")
            print(f"Mean ISOPleasant: {loc_data[col_iso_p].mean():.3f}")
            print(f"Mean ISOEventful: {loc_data[col_iso_e].mean():.3f}\n")
    else:
        print("⚠️ No se pudieron encontrar las columnas para hacer el resumen.")


def generar_radar_circunflejo_paq_global():
    """Genera 3 gráficas Radar Plot Globales: Medias, Medianas y Combinada."""
    print("\n" + "="*50)
    print("GENERANDO RADAR PLOTS GLOBALES (MEDIAS vs MEDIANAS)")
    print("="*50)

    carpeta = obtener_carpeta_guardado("General")
    variables_paq = ['pleasant', 'vibrant', 'eventful', 'chaotic', 'annoying', 'monotonous', 'uneventful', 'calm']
    cols = df_datos.columns.tolist()

    if len([v for v in variables_paq if v in cols]) < 8:
         print("⚠️ ERROR: No se encuentran todos los 8 PAQs necesarios.")
         return

    # --- 1. PREPARACIÓN DE DATOS ---
    angulos = np.linspace(0, 2 * np.pi, len(variables_paq), endpoint=False).tolist()
    angulos += angulos[:1] # Cerramos el círculo

    # Cálculo de Medias
    medias = df_datos[variables_paq].mean().tolist()
    medias += medias[:1]
    
    # Cálculo de Medianas
    medianas = df_datos[variables_paq].median().tolist()
    medianas += medianas[:1]

    # Cálculo de coordenadas ISO (Medias y Medianas)
    col_iso_p = next((c for c in cols if c.lower() == 'isopleasant'), None)
    col_iso_e = next((c for c in cols if c.lower() == 'isoeventful'), None)
    
    iso_p_mean = df_datos[col_iso_p].mean() if col_iso_p else 0
    iso_e_mean = df_datos[col_iso_e].mean() if col_iso_e else 0
    iso_p_median = df_datos[col_iso_p].median() if col_iso_p else 0
    iso_e_median = df_datos[col_iso_e].median() if col_iso_e else 0

    # --- 2. SUB-FUNCIÓN DE DIBUJADO ---
    def dibujar_radar(lista_datos, lista_colores, lista_etiquetas, titulo, nombre_archivo):
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.set_theta_offset(0)
        ax.set_theta_direction(1)

        for datos, color, etiqueta in zip(lista_datos, lista_colores, lista_etiquetas):
            ax.plot(angulos, datos, color=color, linewidth=2, linestyle='solid', label=etiqueta)
            ax.fill(angulos, datos, color=color, alpha=0.2)

        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(variables_paq, fontsize=12)
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['1', '2', '3', '4', '5'], color="grey", size=10)
        
        plt.title(titulo, fontsize=15, pad=30)
        plt.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.1))

        ruta_img = carpeta / nombre_archivo
        plt.savefig(ruta_img, bbox_inches='tight', dpi=300, facecolor='white') 
        plt.close() 
        print(f"  ✅ Guardado: Graficas_Venecia_TFG/General/{ruta_img.name}")

    # --- 3. GENERACIÓN DE LAS 3 GRÁFICAS ---
    tit_mean = f'Circumplex Model (Global Mean)\nISOPleasant: {iso_p_mean:.2f} | ISOEventful: {iso_e_mean:.2f}'
    dibujar_radar([medias], ['#0072B2'], ['Mean (Media)'], tit_mean, '02A_Radar_PAQ_Global_Mean.png')

    tit_median = f'Circumplex Model (Global Median)\nISOPleasant: {iso_p_median:.2f} | ISOEventful: {iso_e_median:.2f}'
    dibujar_radar([medianas], ['#D55E00'], ['Median (Mediana)'], tit_median, '02B_Radar_PAQ_Global_Median.png')

    tit_combo = (f'Circumplex Model (Mean vs Median)\n'
                 f'Mean (P: {iso_p_mean:.2f}, E: {iso_e_mean:.2f}) | Median (P: {iso_p_median:.2f}, E: {iso_e_median:.2f})')
    dibujar_radar([medias, medianas], ['#0072B2', '#D55E00'], ['Mean (Media)', 'Median (Mediana)'], tit_combo, '02C_Radar_PAQ_Global_Combined.png')

    print("\n🎉 ¡ÉXITO! Se han generado las 3 versiones del radar plot global.")


def generar_radar_localizacion_especifica():
    """Genera 3 gráficas Radar Plot por Localización: Medias, Medianas y Combinada."""
    print("\n" + "="*50)
    print("GENERANDO RADAR PLOTS POR LOCALIZACIÓN (MEDIAS vs MEDIANAS)")
    print("="*50)

    variables_paq = ['pleasant', 'vibrant', 'eventful', 'chaotic', 'annoying', 'monotonous', 'uneventful', 'calm']
    cols = df_datos.columns.tolist()
    col_loc = next((c for c in cols if c.lower() == 'locationid'), None)

    if not col_loc:
        print("⚠️ ERROR: No se encuentra la columna 'locationid'.")
        return

    localizaciones = df_datos[col_loc].dropna().unique().tolist()
    
    print("\n📍 Localizaciones disponibles:")
    for loc in localizaciones:
        print(f"  - {loc}")
        
    print("\n💡 Truco: Escribe 'todas' para generar las gráficas de todas las localizaciones a la vez.")
    loc_elegida = input("👉 Escribe el nombre de la localización, 'todas', o '0' para salir: ").strip()
    
    if loc_elegida == '0': return
        
    locs_a_procesar = []
    if loc_elegida.lower() == 'todas':
        locs_a_procesar = localizaciones
        print(f"\nGenerando automáticamente gráficas para {len(localizaciones)} localizaciones...")
    else:
        match = next((l for l in localizaciones if l.lower() == loc_elegida.lower()), None)
        if match:
            locs_a_procesar = [match]
        else:
            print(f"❌ ERROR: '{loc_elegida}' no está en la lista. Revisa cómo se escribe.")
            return

    col_iso_p = next((c for c in cols if c.lower() == 'isopleasant'), None)
    col_iso_e = next((c for c in cols if c.lower() == 'isoeventful'), None)

    angulos = np.linspace(0, 2 * np.pi, len(variables_paq), endpoint=False).tolist()
    angulos += angulos[:1]

    def dibujar_radar_loc(lista_datos, lista_colores, lista_etiquetas, titulo, nombre_archivo, carpeta_dest):
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.set_theta_offset(0)
        ax.set_theta_direction(1)

        for datos, color, etiqueta in zip(lista_datos, lista_colores, lista_etiquetas):
            ax.plot(angulos, datos, color=color, linewidth=2, linestyle='solid', label=etiqueta)
            ax.fill(angulos, datos, color=color, alpha=0.2)

        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(variables_paq, fontsize=12)
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['1', '2', '3', '4', '5'], color="grey", size=10)
        
        plt.title(titulo, fontsize=15, pad=30)
        plt.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.1))

        ruta_img = carpeta_dest / nombre_archivo
        plt.savefig(ruta_img, bbox_inches='tight', dpi=300, facecolor='white') 
        plt.close()

    for loc in locs_a_procesar:
        df_loc = df_datos[df_datos[col_loc] == loc]
        carpeta = obtener_carpeta_guardado(loc)
        
        medias = df_loc[variables_paq].mean().tolist()
        medias += medias[:1] 
        
        medianas = df_loc[variables_paq].median().tolist()
        medianas += medianas[:1]

        iso_p_mean = df_loc[col_iso_p].mean() if col_iso_p else 0
        iso_e_mean = df_loc[col_iso_e].mean() if col_iso_e else 0
        iso_p_median = df_loc[col_iso_p].median() if col_iso_p else 0
        iso_e_median = df_loc[col_iso_e].median() if col_iso_e else 0

        tit_mean = f'Circumplex Model (Mean) - {loc}\nISOPleasant: {iso_p_mean:.2f} | ISOEventful: {iso_e_mean:.2f}'
        dibujar_radar_loc([medias], ['#E69F00'], [f'Mean - {loc}'], tit_mean, f'03A_Radar_PAQ_Mean_{loc}.png', carpeta)

        tit_median = f'Circumplex Model (Median) - {loc}\nISOPleasant: {iso_p_median:.2f} | ISOEventful: {iso_e_median:.2f}'
        dibujar_radar_loc([medianas], ['#009E73'], [f'Median - {loc}'], tit_median, f'03B_Radar_PAQ_Median_{loc}.png', carpeta)

        tit_combo = (f'Circumplex Model (Mean vs Median) - {loc}\n'
                     f'Mean (P: {iso_p_mean:.2f}, E: {iso_e_mean:.2f}) | Median (P: {iso_p_median:.2f}, E: {iso_e_median:.2f})')
        dibujar_radar_loc([medias, medianas], ['#E69F00', '#009E73'], ['Mean (Media)', 'Median (Mediana)'], tit_combo, f'03C_Radar_PAQ_Combined_{loc}.png', carpeta)
        
        print(f"✅ 3 Gráficos guardados para {loc} en Graficas_Venecia_TFG/{loc}")

    print(f"\n🎉 Proceso finalizado.")


def generar_iso_plot_combinado():
    print("\n" + "="*50)
    print("GENERANDO VISUALIZACIÓN COMBINADA (DENSIDAD + PUNTOS)")
    print("="*50)

    cols = df_datos.columns.tolist()
    col_loc = next((c for c in cols if c.lower() == 'locationid'), None)
    col_iso_p = next((c for c in cols if c.lower() == 'isopleasant'), None)
    col_iso_e = next((c for c in cols if c.lower() == 'isoeventful'), None)

    if not col_loc or not col_iso_p or not col_iso_e:
        print("⚠️ ERROR: No se encuentran las columnas necesarias ('locationid', 'isopleasant', 'isoeventful').")
        return

    localizaciones = df_datos[col_loc].dropna().unique().tolist()
    
    print("\n📍 Localizaciones disponibles:")
    for loc in localizaciones:
        print(f"  - {loc}")
        
    print("\n💡 Truco: Escribe 'todas' para generar las gráficas de todas las localizaciones a la vez.")
    loc_elegida = input("👉 Escribe el nombre de la localización, 'todas', o '0' para salir: ").strip()
    
    if loc_elegida == '0': return
        
    locs_a_procesar = []
    if loc_elegida.lower() == 'todas':
        locs_a_procesar = localizaciones
        print(f"\nGenerando automáticamente {len(localizaciones)} gráficas combinadas...")
    else:
        match = next((l for l in localizaciones if l.lower() == loc_elegida.lower()), None)
        if match:
            locs_a_procesar = [match]
        else:
            print(f"❌ ERROR: '{loc_elegida}' no está en la lista. Revisa cómo se escribe.")
            return

    for loc in locs_a_procesar:
        df_loc = df_datos[df_datos[col_loc] == loc]

        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            sns.kdeplot(data=df_loc, x=col_iso_p, y=col_iso_e, fill=True, cmap="Blues", alpha=0.6, levels=8, ax=ax)
            sns.scatterplot(data=df_loc, x=col_iso_p, y=col_iso_e, color="steelblue", edgecolor="black", s=30, alpha=0.7, ax=ax)
            dar_formato_ejes_iso(ax, f"Combined Visualization for {loc}")
            
            carpeta = obtener_carpeta_guardado(loc)
            ruta_img = carpeta / f'ISO_Combined_Plot_{loc}.png'
            plt.savefig(ruta_img, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close()
            
            print(f"✅ Gráfico combinado guardado: Graficas_Venecia_TFG/{loc}/{ruta_img.name}")
            
        except Exception as e:
            print(f"❌ Error al generar el gráfico para {loc}: {e}")
            
    print(f"\n🎉 Proceso finalizado.")


def calcular_parametros_estatisticos_nativos():
    print("\n" + "="*50)
    print("CÁLCULO SPI Y GRÁFICA COMPARATIVA: REALIDAD vs OBJETIVO")
    print("="*50)

    cols = df_datos.columns.tolist()
    col_loc = next((c for c in cols if c.lower() == 'locationid'), None)
    col_iso_p = next((c for c in cols if c.lower() == 'isopleasant'), None)
    col_iso_e = next((c for c in cols if c.lower() == 'isoeventful'), None)

    if not col_loc or not col_iso_p or not col_iso_e:
        print("⚠️ ERROR: No se encuentran las columnas necesarias ('locationid', 'isopleasant', 'isoeventful').")
        return

    localizaciones = df_datos[col_loc].dropna().unique().tolist()
    
    print("\n📍 Localizaciones disponibles:")
    for loc in localizaciones:
        print(f"  - {loc}")

    while True:
        print("\n" + "-"*40)
        loc_elegida = input("👉 Escribe el nombre de una localización, 'todas', o '0' para volver al menú: ").strip()
        
        if loc_elegida == '0':
            print("Volviendo al menú principal...")
            break 
            
        locs_a_procesar = []
        if loc_elegida.lower() == 'todas':
            locs_a_procesar = localizaciones
        else:
            match = next((l for l in localizaciones if l.lower() == loc_elegida.lower()), None)
            if match:
                locs_a_procesar = [match]
            else:
                print(f"❌ ERROR: '{loc_elegida}' no está en la lista. Revisa cómo se escribe e inténtalo de nuevo.")
                continue

        print("\n🎯 ¿Cuál es la 'Target Distribution' (Objetivo ideal) para este análisis?")
        print("1. Vibrant (Agradable + Animado) -> Ej: Calles comerciales, terrazas.")
        print("2. Calm (Agradable + Inactivo) -> Ej: Parques, zonas de descanso.")
        print("3. Pleasant (Agradable Neutro) -> Ej: Plazas generales.")
        print("4. Eventful (Animado Neutro) -> Ej: Mercados, ferias.")
        
        opcion_target = input("\nElige el objetivo (1-4): ").strip()
        
        dict_targets = {
            '1': {"nombre": "Vibrant", "mean": [0.6, 0.6]},
            '2': {"nombre": "Calm", "mean": [0.6, -0.6]},
            '3': {"nombre": "Pleasant", "mean": [0.8, 0.0]},
            '4': {"nombre": "Eventful", "mean": [0.0, 0.8]}
        }
        
        target_elegido = dict_targets.get(opcion_target, dict_targets['3'])
        target_mean = np.array(target_elegido["mean"])
        target_cov = np.array([[0.1, 0.0], [0.0, 0.1]]) 
        
        print(f"\n✅ Objetivo fijado a: {target_elegido['nombre']} (Centro teórico: P={target_mean[0]}, E={target_mean[1]})")
        print("Generando análisis...\n")

        for loc in locs_a_procesar:
            try:
                data_loc = df_datos[df_datos[col_loc] == loc]
                p_data = data_loc[col_iso_p]
                e_data = data_loc[col_iso_e]
                
                mean_p_real = p_data.mean()
                mean_e_real = e_data.mean()
                real_mean = np.array([mean_p_real, mean_e_real])
                cov_real = data_loc[[col_iso_p, col_iso_e]].cov().values
                skew_p = p_data.skew()
                skew_e = e_data.skew()

                distancia = np.linalg.norm(real_mean - target_mean)
                max_dist = 2.8 
                spi_score = max(0, min(100, 100 * (1 - (distancia / max_dist))))

                print(f"Empirical Statistical parameters for {loc}:")
                print("Centred Parameters (Real):")
                print(f"mean:   [{mean_p_real:.3f} {mean_e_real:.3f}]")
                print(f"sigma:  [[ {cov_real[0][0]:.3f}  {cov_real[0][1]:.3f}]\n"
                      f"         [ {cov_real[1][0]:.3f}  {cov_real[1][1]:.3f}]]")
                print(f"skew:   [{skew_p:.3f} {skew_e:.3f}]")
                print("-" * 40)
                print(f"SPI between {loc} and target '{target_elegido['nombre']}': {int(spi_score)}")
                print("-" * 40)

                respuesta_grafica = input(f"👉 ¿Deseas generar y guardar el gráfico doble (Azul vs Rojo) para {loc}? (s/n): ").strip().lower()
                puntos_target = np.random.multivariate_normal(target_mean, target_cov, size=2000)
                df_target = pd.DataFrame(puntos_target, columns=['P_sim', 'E_sim'])

                if respuesta_grafica == 's':
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                    sns.kdeplot(data=data_loc, x=col_iso_p, y=col_iso_e, fill=True, cmap="Blues", alpha=0.6, levels=8, ax=ax1)
                    sns.scatterplot(data=data_loc, x=col_iso_p, y=col_iso_e, color="steelblue", edgecolor="black", s=20, alpha=0.7, ax=ax1)
                    dar_formato_ejes_iso(ax1, f"Real Soundscape Density\n{loc}")

                    sns.kdeplot(data=df_target, x='P_sim', y='E_sim', fill=True, cmap="Reds", alpha=0.6, levels=8, ax=ax2)
                    dar_formato_ejes_iso(ax2, f"Target Distribution: '{target_elegido['nombre']}'\n(SPI Score: {int(spi_score)}/100)")

                    plt.tight_layout()
                    carpeta = obtener_carpeta_guardado(loc)
                    ruta_img = carpeta / f'SPI_Target_{target_elegido["nombre"]}_{loc}.png'
                    plt.savefig(ruta_img, bbox_inches='tight', dpi=300, facecolor='white')
                    plt.close()
                    print(f"✅ Gráfica comparativa SPI guardada: Graficas_Venecia_TFG/{loc}/{ruta_img.name}\n")
                else:
                    print(f"⏭️ Omitiendo gráfica doble para {loc}.\n")

                respuesta_superpuesta = input(f"👉 ¿Deseas generar el gráfico con ambas áreas superpuestas para {loc}? (s/n): ").strip().lower()
                
                if respuesta_superpuesta == 's':
                    fig, ax = plt.subplots(figsize=(8, 8))
                    
                    sns.kdeplot(data=data_loc, x=col_iso_p, y=col_iso_e, fill=True, cmap="Blues", alpha=0.6, levels=8, ax=ax)
                    sns.scatterplot(data=data_loc, x=col_iso_p, y=col_iso_e, color="steelblue", edgecolor="black", s=15, alpha=0.5, ax=ax)
                    
                    sns.kdeplot(data=df_target, x='P_sim', y='E_sim', fill=True, color="red", alpha=0.4, levels=[0.1, 1], ax=ax)
                    sns.kdeplot(data=df_target, x='P_sim', y='E_sim', color="red", linewidths=2, levels=[0.1], ax=ax)

                    dar_formato_ejes_iso(ax, f"Comparison of Soundscape Perceptions against a {target_elegido['nombre']} target\n{loc}")

                    props_bbox = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray")
                    ax.text(0, -0.9, f"SPI: {int(spi_score)}", fontsize=12, ha='center', va='center', bbox=props_bbox)

                    carpeta = obtener_carpeta_guardado(loc)
                    ruta_img_super = carpeta / f'SPI_Comparison_{target_elegido["nombre"]}_{loc}.png'
                    plt.savefig(ruta_img_super, bbox_inches='tight', dpi=300, facecolor='white')
                    plt.close()
                    print(f"✅ Gráfica superpuesta guardada: Graficas_Venecia_TFG/{loc}/{ruta_img_super.name}\n")
                else:
                    print(f"⏭️ Omitiendo gráfica superpuesta para {loc}.\n")

            except Exception as e:
                print(f"❌ Error al calcular/graficar para {loc}: {e}\n")

        print("🎉 Ronda de análisis finalizada. Puedes elegir otra localización.")

def generar_scatter_agrupado_localizacion():
    print("\n" + "="*50)
    print("GENERANDO SCATTER PLOT AGRUPADO POR LOCALIZACIÓN")
    print("="*50)

    cols = df_datos.columns.tolist()
    col_loc = next((c for c in cols if c.lower() == 'locationid'), None)
    col_iso_p = next((c for c in cols if c.lower() == 'isopleasant'), None)
    col_iso_e = next((c for c in cols if c.lower() == 'isoeventful'), None)

    if not col_loc or not col_iso_p or not col_iso_e:
        print("⚠️ ERROR: No se encuentran las columnas necesarias ('locationid', 'isopleasant', 'isoeventful').")
        return

    carpeta = obtener_carpeta_guardado("General")

    try:
        fig, ax = plt.subplots(figsize=(10, 10))

        sns.scatterplot(
            data=df_datos,
            x=col_iso_p,
            y=col_iso_e,
            hue=col_loc,
            palette="turbo",     
            alpha=0.9,          
            s=60,               
            edgecolor='black',  
            linewidth=0.5,
            ax=ax
        )

        dar_formato_ejes_iso(ax, "Scatter Plot Grouped by Location")
        plt.legend(title="LocationID", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', markerscale=1.5)
        plt.tight_layout()

        ruta_img = carpeta / '06_Scatter_Grouped_by_Location.png'
        plt.savefig(ruta_img, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        print(f"✅ Gráfico scatter agrupado guardado: Graficas_Venecia_TFG/General/{ruta_img.name}")

    except Exception as e:
        print(f"❌ Error al generar el gráfico scatter agrupado: {e}")


def generar_iso_plots_individuales_por_sesion():
    print("\n" + "="*50)
    print("GENERANDO GRÁFICOS ISO AGRUPADOS POR SESIÓN")
    print("="*50)

    cols = df_datos.columns.tolist()
    col_loc = next((c for c in cols if c.lower() == 'locationid'), None)
    col_session = next((c for c in cols if c.lower() == 'sessionid'), None)
    col_iso_p = next((c for c in cols if c.lower() == 'isopleasant'), None)
    col_iso_e = next((c for c in cols if c.lower() == 'isoeventful'), None)

    if not col_loc or not col_session or not col_iso_p or not col_iso_e:
        print("⚠️ ERROR: Faltan columnas necesarias ('locationid', 'sessionid', 'isopleasant', 'isoeventful').")
        return

    localizaciones = df_datos[col_loc].dropna().unique().tolist()
    
    print("\n📍 Localizaciones disponibles:")
    for loc in localizaciones:
        print(f"  - {loc}")

    while True:
        print("\n" + "-"*40)
        loc_elegida = input("👉 Escribe la localización, 'todas', o '0' para volver al menú: ").strip()
        
        if loc_elegida == '0':
            print("Volviendo al menú principal...")
            break 
            
        locs_a_procesar = []
        if loc_elegida.lower() == 'todas':
            locs_a_procesar = localizaciones
        else:
            match = next((l for l in localizaciones if l.lower() == loc_elegida.lower()), None)
            if match:
                locs_a_procesar = [match]
            else:
                print(f"❌ ERROR: '{loc_elegida}' no está en la lista. Inténtalo de nuevo.")
                continue

        for loc in locs_a_procesar:
            df_loc = df_datos[df_datos[col_loc] == loc]
            sesiones = df_loc[col_session].dropna().unique().tolist()
            
            sesiones_validas = []
            for s in sesiones:
                if len(df_loc[df_loc[col_session] == s]) >= 3:
                    sesiones_validas.append(s)
                else:
                    print(f"  ⚠️ Sesión '{s}' en {loc} omitida por falta de datos suficientes para densidad.")

            if not sesiones_validas:
                print(f"  ⏭️ No hay suficientes datos válidos por sesión para {loc}. Saltando...")
                continue
                
            df_loc_valido = df_loc[df_loc[col_session].isin(sesiones_validas)]

            try:
                fig, ax = plt.subplots(figsize=(10, 8))
                paleta = sns.color_palette("husl", len(sesiones_validas))

                sns.scatterplot(
                    data=df_loc_valido, 
                    x=col_iso_p, 
                    y=col_iso_e, 
                    hue=col_session, 
                    palette=paleta,
                    edgecolor="white",
                    linewidth=0.5,
                    s=40, 
                    alpha=0.8, 
                    ax=ax
                )
                
                sns.kdeplot(
                    data=df_loc_valido, 
                    x=col_iso_p, 
                    y=col_iso_e, 
                    hue=col_session, 
                    palette=paleta,
                    fill=False,
                    levels=3,
                    linewidths=2,
                    alpha=0.9,
                    legend=False, 
                    ax=ax
                )
                
                dar_formato_ejes_iso(ax, f"Soundscape Perception Grouped by Session\nLocation: {loc}")
                ax.legend(title="SessionID", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', markerscale=1.5)
                
                plt.tight_layout()
                carpeta = obtener_carpeta_guardado(loc)
                ruta_img = carpeta / f'ISO_Plot_All_Sessions_{loc}.png'
                plt.savefig(ruta_img, bbox_inches='tight', dpi=300, facecolor='white')
                plt.close()
                print(f"  ✅ Gráfico agrupado de sesiones guardado: Graficas_Venecia_TFG/{loc}/{ruta_img.name}")
                
            except Exception as e:
                print(f"  ❌ Error al procesar {loc}: {e}")
                
        print("\n🎉 Ronda de análisis finalizada. Puedes elegir otra localización.")

def generar_scatter_medias_localizaciones():
    """Calcula la media y mediana ISO de cada localización y las representa en Scatter Plots."""
    print("\n" + "="*50)
    print("GENERANDO SCATTER PLOTS (MEDIAS vs MEDIANAS) POR LOCALIZACIÓN")
    print("="*50)

    cols = df_datos.columns.tolist()
    col_loc = next((c for c in cols if c.lower() == 'locationid'), None)
    col_iso_p = next((c for c in cols if c.lower() == 'isopleasant'), None)
    col_iso_e = next((c for c in cols if c.lower() == 'isoeventful'), None)

    if not col_loc or not col_iso_p or not col_iso_e:
        print("⚠️ ERROR: No se encuentran las columnas necesarias ('locationid', 'isopleasant', 'isoeventful').")
        return

    carpeta = obtener_carpeta_guardado("General")

    try:
        df_medias = df_datos.groupby(col_loc)[[col_iso_p, col_iso_e]].mean().reset_index()
        df_medianas = df_datos.groupby(col_loc)[[col_iso_p, col_iso_e]].median().reset_index()

        df_medias = df_medias.sort_values(by=col_loc).reset_index(drop=True)
        df_medianas = df_medianas.sort_values(by=col_loc).reset_index(drop=True)

        def dibujar_scatter(tipo, titulo, archivo):
            fig, ax = plt.subplots(figsize=(10, 10))

            if tipo in ['mean', 'combined']:
                sns.scatterplot(
                    data=df_medias, x=col_iso_p, y=col_iso_e,
                    color='#E69F00', s=70, marker='o', ax=ax, zorder=5,
                    label='Mean (Media)' if tipo == 'combined' else None
                )
                for idx, row in df_medias.iterrows():
                    ax.annotate(str(row[col_loc]), (row[col_iso_p], row[col_iso_e]),
                                xytext=(8, 5), textcoords="offset points",
                                fontsize=10, fontweight="bold", color="#1a1a1a")

            if tipo in ['median', 'combined']:
                sns.scatterplot(
                    data=df_medianas, x=col_iso_p, y=col_iso_e,
                    color='#009E73', s=60, marker='s' if tipo == 'combined' else 'o', ax=ax, zorder=6,
                    label='Median (Mediana)' if tipo == 'combined' else None
                )
                if tipo == 'median':
                    for idx, row in df_medianas.iterrows():
                        ax.annotate(str(row[col_loc]), (row[col_iso_p], row[col_iso_e]),
                                    xytext=(8, 5), textcoords="offset points",
                                    fontsize=10, fontweight="bold", color="#1a1a1a")

            if tipo == 'combined':
                for idx in range(len(df_medias)):
                    x_val = [df_medias.iloc[idx][col_iso_p], df_medianas.iloc[idx][col_iso_p]]
                    y_val = [df_medias.iloc[idx][col_iso_e], df_medianas.iloc[idx][col_iso_e]]
                    ax.plot(x_val, y_val, color='gray', linestyle='--', linewidth=1, zorder=4, alpha=0.6)

                ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12, markerscale=1.5)

            dar_formato_ejes_iso(ax, titulo)
            plt.tight_layout()

            ruta_img = carpeta / archivo
            plt.savefig(ruta_img, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close()
            print(f"  ✅ Guardado: Graficas_Venecia_TFG/General/{ruta_img.name}")


        dibujar_scatter('mean', "Mean ISO Coordinates by Location", '08A_Scatter_Mean_ISO_by_Location.png')
        dibujar_scatter('median', "Median ISO Coordinates by Location", '08B_Scatter_Median_ISO_by_Location.png')
        dibujar_scatter('combined', "Mean vs Median ISO Coordinates by Location", '08C_Scatter_Combined_ISO_by_Location.png')

        print("\n🎉 ¡ÉXITO! Se han generado las 3 versiones del Scatter Plot en la carpeta General.")

    except Exception as e:
        print(f"❌ Error al generar los gráficos de medias/medianas: {e}")

def generar_jointplot_localizacion():
    """Genera un Joint Plot (Densidad/Puntos + Histogramas) individual por localización."""
    print("\n" + "="*50)
    print("GENERANDO JOINT PLOTS (DENSIDAD + HISTOGRAMAS)")
    print("="*50)

    cols = df_datos.columns.tolist()
    col_loc = next((c for c in cols if c.lower() == 'locationid'), None)
    col_iso_p = next((c for c in cols if c.lower() == 'isopleasant'), None)
    col_iso_e = next((c for c in cols if c.lower() == 'isoeventful'), None)

    if not col_loc or not col_iso_p or not col_iso_e:
        print("⚠️ ERROR: No se encuentran las columnas necesarias ('locationid', 'isopleasant', 'isoeventful').")
        return

    localizaciones = df_datos[col_loc].dropna().unique().tolist()
    
    print("\n📍 Localizaciones disponibles:")
    for loc in localizaciones:
        print(f"  - {loc}")

    while True:
        print("\n" + "-"*40)
        loc_elegida = input("👉 Escribe la localización, 'todas', o '0' para volver al menú: ").strip()
        
        if loc_elegida == '0':
            print("Volviendo al menú principal...")
            break 
            
        locs_a_procesar = []
        if loc_elegida.lower() == 'todas':
            locs_a_procesar = localizaciones
            print(f"\nGenerando automáticamente {len(localizaciones)} Joint Plots...")
        else:
            match = next((l for l in localizaciones if l.lower() == loc_elegida.lower()), None)
            if match:
                locs_a_procesar = [match]
            else:
                print(f"❌ ERROR: '{loc_elegida}' no está en la lista. Revisa cómo se escribe e inténtalo de nuevo.")
                continue

        for loc in locs_a_procesar:
            df_loc = df_datos[df_datos[col_loc] == loc]

            try:
                g = sns.JointGrid(data=df_loc, x=col_iso_p, y=col_iso_e, height=8, space=0.2)

                sns.kdeplot(data=df_loc, x=col_iso_p, y=col_iso_e, fill=True, cmap="Blues", alpha=0.6, levels=8, ax=g.ax_joint)
                sns.scatterplot(data=df_loc, x=col_iso_p, y=col_iso_e, color="steelblue", edgecolor="black", s=30, alpha=0.7, ax=g.ax_joint)

                sns.histplot(data=df_loc, x=col_iso_p, ax=g.ax_marg_x, fill=True, color="steelblue", bins=15, edgecolor="white", alpha=0.8)
                sns.histplot(data=df_loc, y=col_iso_e, ax=g.ax_marg_y, fill=True, color="steelblue", bins=15, edgecolor="white", alpha=0.8)

                dar_formato_ejes_iso(g.ax_joint, "")
                
                g.fig.suptitle(f"{loc} - Joint Plot with Histograms", y=1.02, fontsize=16)

                carpeta = obtener_carpeta_guardado(loc)
                ruta_img = carpeta / f'09_JointPlot_{loc}.png'
                g.savefig(ruta_img, bbox_inches='tight', dpi=300, facecolor='white')
                plt.close(g.fig) 
                
                print(f"✅ Joint Plot guardado: Graficas_Venecia_TFG/{loc}/{ruta_img.name}")
                
            except Exception as e:
                print(f"❌ Error al generar el Joint Plot para {loc}: {e}")
                
        print("🎉 Ronda de análisis finalizada. Puedes elegir otra localización.")


def generar_jointplot_agrupado_localizaciones():
    """Genera un Joint Plot agrupado (Densidad central y marginal) seleccionando localizaciones a la carta."""
    print("\n" + "="*50)
    print("GENERANDO JOINT PLOT AGRUPADO (COMPARATIVA)")
    print("="*50)

    cols = df_datos.columns.tolist()
    col_loc = next((c for c in cols if c.lower() == 'locationid'), None)
    col_iso_p = next((c for c in cols if c.lower() == 'isopleasant'), None)
    col_iso_e = next((c for c in cols if c.lower() == 'isoeventful'), None)

    if not col_loc or not col_iso_p or not col_iso_e:
        print("⚠️ ERROR: No se encuentran las columnas necesarias ('locationid', 'isopleasant', 'isoeventful').")
        return

    localizaciones_disponibles = df_datos[col_loc].dropna().unique().tolist()
    
    print("\n📍 Localizaciones disponibles:")
    for loc in localizaciones_disponibles:
        print(f"  - {loc}")

    locs_elegidas = []
    print("\n💡 INSTRUCCIONES: Escribe el nombre de las zonas una a una.")
    print("   - Cuando tengas todas las que quieres comparar, escribe 'fin'.")
    print("   - Escribe '0' para cancelar y salir al menú principal.")
    
    while True:
        entrada = input(f"👉 Localización {len(locs_elegidas) + 1} (o 'fin' / '0'): ").strip()
        
        if entrada == '0':
            print("Operación cancelada. Volviendo al menú...")
            return
        elif entrada.lower() == 'fin':
            if len(locs_elegidas) == 0:
                print("⚠️ No has elegido ninguna localización. Volviendo al menú...")
                return
            else:
                break 
        else:
            match = next((l for l in localizaciones_disponibles if l.lower() == entrada.lower()), None)
            if match:
                if match not in locs_elegidas:
                    locs_elegidas.append(match)
                    print(f"  ✅ '{match}' añadida a la comparativa.")
                else:
                    print("  ⚠️ Esa localización ya está en la lista.")
            else:
                print(f"  ❌ ERROR: '{entrada}' no existe. Revisa cómo se escribe.")

    print(f"\nGenerando Joint Plot comparativo para {len(locs_elegidas)} localizaciones...")
    
    df_filtrado = df_datos[df_datos[col_loc].isin(locs_elegidas)]

    try:
        sns.set_style("darkgrid") 
        paleta = sns.color_palette("bright", len(locs_elegidas))
        g = sns.JointGrid(data=df_filtrado, x=col_iso_p, y=col_iso_e, hue=col_loc, palette=paleta, height=8, space=0)

        sns.scatterplot(data=df_filtrado, x=col_iso_p, y=col_iso_e, hue=col_loc, palette=paleta, 
                        edgecolor="black", s=40, alpha=0.8, ax=g.ax_joint, legend="full")
        
        sns.kdeplot(data=df_filtrado, x=col_iso_p, y=col_iso_e, hue=col_loc, palette=paleta, 
                    fill=True, alpha=0.2, levels=[0.1, 1], ax=g.ax_joint, legend=False)
        
        sns.kdeplot(data=df_filtrado, x=col_iso_p, y=col_iso_e, hue=col_loc, palette=paleta, 
                    fill=False, linewidths=2, levels=[0.1], ax=g.ax_joint, legend=False)

        sns.kdeplot(data=df_filtrado, x=col_iso_p, hue=col_loc, palette=paleta, fill=True, alpha=0.4, 
                    ax=g.ax_marg_x, legend=False, common_norm=False)
        sns.kdeplot(data=df_filtrado, y=col_iso_e, hue=col_loc, palette=paleta, fill=True, alpha=0.4, 
                    ax=g.ax_marg_y, legend=False, common_norm=False)

        dar_formato_ejes_iso(g.ax_joint, "")
        g.fig.suptitle("Comparative Joint Plot", y=1.02, fontsize=16)
        
        g.ax_joint.legend(title="LocationID", bbox_to_anchor=(1.25, 0.8), loc='upper left', 
                          frameon=True, facecolor='white', framealpha=1, markerscale=1.5)

        carpeta = obtener_carpeta_guardado("General")
        nombres_cortos = "_".join([loc[:4] for loc in locs_elegidas]) 
        ruta_img = carpeta / f'10_JointPlot_Grouped_{nombres_cortos}.png'
        
        g.savefig(ruta_img, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close(g.fig) 
        
        print(f"✅ Joint Plot comparativo guardado: Graficas_Venecia_TFG/General/{ruta_img.name}")
        
    except Exception as e:
        print(f"❌ Error al generar el Joint Plot agrupado: {e}")

def resumir_datos_localizaciones():
    """Genera tablas resumen (count, medias y medianas) de los PAQs e ISO, y permite exportar o consultar."""
    print("\n" + "="*50)
    print("RESUMEN ESTADÍSTICO SUBJETIVO POR LOCALIZACIÓN (DESCRIBE)")
    print("="*50)

    cols = df_datos.columns.tolist()
    col_loc = next((c for c in cols if c.lower() == 'locationid'), None)
    
    vars_buscadas = ['isopleasant', 'isoeventful', 'pleasant', 'vibrant', 'eventful', 
                     'chaotic', 'annoying', 'monotonous', 'uneventful', 'calm']
    cols_num_reales = [c for c in cols if c.lower() in vars_buscadas]
    
    if not col_loc or not cols_num_reales:
        print("⚠️ ERROR: Faltan columnas necesarias para hacer el resumen.")
        return

    df_count = df_datos.groupby(col_loc).size().reset_index(name='count')
    
    df_means_calc = df_datos.groupby(col_loc)[cols_num_reales].mean().reset_index()
    df_resumen_medias = pd.merge(df_count, df_means_calc, on=col_loc).round(3)
    
    df_medians_calc = df_datos.groupby(col_loc)[cols_num_reales].median().reset_index()
    df_resumen_medianas = pd.merge(df_count, df_medians_calc, on=col_loc).round(3)

    while True:
        print("\n" + "-"*40)
        print("¿Qué deseas hacer con este resumen de datos?")
        print("1. Guardar CSV con las MEDIAS por localización")
        print("2. Guardar CSV con las MEDIANAS por localización")
        print("3. Consultar datos (Medias y Medianas) de una localización por terminal")
        print("0. Volver al menú principal")
        
        sub_op = input("\nElige una opción (0-3): ").strip()
        
        if sub_op == '1':
            carpeta = obtener_carpeta_guardado("General")
            ruta_csv = carpeta / "11A_Resumen_Subjetivo_Venecia_Medias.csv"
            df_resumen_medias.to_csv(ruta_csv, index=False, sep=';', decimal=',')
            print(f"✅ ¡CSV de MEDIAS guardado con éxito en: Graficas_Venecia_TFG/General/{ruta_csv.name}!")
            
        elif sub_op == '2':
            carpeta = obtener_carpeta_guardado("General")
            ruta_csv = carpeta / "11B_Resumen_Subjetivo_Venecia_Medianas.csv"
            df_resumen_medianas.to_csv(ruta_csv, index=False, sep=';', decimal=',')
            print(f"✅ ¡CSV de MEDIANAS guardado con éxito en: Graficas_Venecia_TFG/General/{ruta_csv.name}!")
            
        elif sub_op == '3':
            localizaciones_disponibles = df_resumen_medias[col_loc].tolist()
            print("\n📍 Localizaciones disponibles:")
            for loc in localizaciones_disponibles:
                print(f"  - {loc}")
                
            while True:
                loc_elegida = input("\n👉 Escribe la localización para ver sus datos (o '0' para salir al submenú): ").strip()
                
                if loc_elegida == '0':
                    break 
                    
                match = next((l for l in localizaciones_disponibles if l.lower() == loc_elegida.lower()), None)
                if match:
                    loc_media = df_resumen_medias[df_resumen_medias[col_loc] == match].iloc[0]
                    loc_mediana = df_resumen_medianas[df_resumen_medianas[col_loc] == match].iloc[0]
                    
                    print(f"\n📊 Datos para: {match} (Encuestas: {loc_media['count']})")
                    print("-" * 50)
                    print(f"{'VARIABLE'.ljust(15)} | {'MEDIA'.center(12)} | {'MEDIANA'.center(12)}")
                    print("-" * 50)
                    
                    for col in cols_num_reales:
                        var_name = col.capitalize().ljust(15)
                        val_media = f"{loc_media[col]:.3f}".center(12)
                        val_mediana = f"{loc_mediana[col]:.3f}".center(12)
                        print(f"{var_name} | {val_media} | {val_mediana}")
                    print("-" * 50)
                else:
                    print(f"❌ ERROR: '{loc_elegida}' no existe. Revisa cómo se escribe.")
        
        elif sub_op == '0':
            print("Saliendo de la herramienta de resumen...")
            break
        else:
            print("❌ Opción no válida. Por favor, elige 0, 1, 2 o 3.")

def resumir_caracteristicas_acusticas():
    """Genera resúmenes y extracciones de las características físicas y psicoacústicas del sonido (Medias y Medianas)."""
    print("\n" + "="*50)
    print("ANÁLISIS DE CARACTERÍSTICAS ACÚSTICAS (CRUDO, MEDIAS Y MEDIANAS)")
    print("="*50)

    cols = df_datos.columns.tolist()
    col_loc = next((c for c in cols if c.lower() == 'locationid'), None)
    
    if not col_loc:
        print("⚠️ ERROR: No se encuentra la columna 'locationid'.")
        return

    excluir = ['latitude', 'longitude', 'language', 'locationid', 'recordid', 'sessionid', 'start_time', 'end_time']
    prefijos_db = ('lz', 'lc', 'la')
    psico_base = ['n', 's', 'r', 't', 'fs', 'i', 'sil', 'ra', 'thd']
    
    cols_acusticas_brutas = []
    for c in cols:
        cl = str(c).lower().strip() 
        if cl in excluir:
            continue
            
        es_db = cl.startswith(prefijos_db)
        
        es_psico = False
        for p in psico_base:
            if cl == p or cl.startswith(f"{p}_") or (cl.startswith(p) and cl[len(p):].replace('.', '').isdigit()):
                es_psico = True
                break
                
        if es_db or es_psico:
            cols_acusticas_brutas.append(c)
            
    cols_acusticas_validas = []
    for col in cols_acusticas_brutas:
        if df_datos[col].dtype == 'object':
            df_datos[col] = df_datos[col].astype(str).str.replace(',', '.', regex=False)
        
        df_datos[col] = pd.to_numeric(df_datos[col], errors='coerce')
        
        if not df_datos[col].isna().all():
            cols_acusticas_validas.append(col)

    if not cols_acusticas_validas:
        print("⚠️ ERROR: No se han encontrado columnas acústicas válidas en tus datos limpios.")
        return

    col_sess = next((c for c in cols if c.lower() == 'sessionid'), None)
    col_rec = next((c for c in cols if c.lower() == 'recordid'), None)
    
    cols_identificadoras = [col_loc]
    if col_sess: cols_identificadoras.append(col_sess)
    if col_rec: cols_identificadoras.append(col_rec)
    
    df_completo_acustico = df_datos[cols_identificadoras + cols_acusticas_validas]

    df_count = df_datos.groupby(col_loc).size().reset_index(name='count')
    
    df_means_calc = df_datos.groupby(col_loc)[cols_acusticas_validas].mean().reset_index()
    df_resumen_medias = pd.merge(df_count, df_means_calc, on=col_loc).round(3)
    
    df_medians_calc = df_datos.groupby(col_loc)[cols_acusticas_validas].median().reset_index()
    df_resumen_medianas = pd.merge(df_count, df_medians_calc, on=col_loc).round(3)

    while True:
        print("\n" + "-"*40)
        print(f"Se han detectado {len(cols_acusticas_validas)} parámetros acústicos válidos.")
        print(f"Variables capturadas: {', '.join(cols_acusticas_validas)}")
        print("\n¿Qué deseas hacer?")
        print("1. Guardar CSV ACÚSTICO (Todas las filas, pero SOLO variables acústicas)")
        print("2. Guardar CSV MEDIAS (Medias acústicas exactas por localización)")
        print("3. Guardar CSV MEDIANAS (Medianas acústicas exactas por localización)")
        print("4. Consultar datos (Medias y Medianas) de una localización por terminal")
        print("0. Volver al menú principal")
        
        sub_op = input("\nElige una opción (0-4): ").strip()
        
        if sub_op == '1':
            carpeta = obtener_carpeta_guardado("General")
            ruta_csv_1 = carpeta / "12A_Datos_Acusticos_Venecia_Filas_Crudas.csv"
            df_completo_acustico.to_csv(ruta_csv_1, index=False, sep=';', decimal=',')
            print(f"✅ ¡CSV ACÚSTICO (todas las filas) guardado en: Graficas_Venecia_TFG/General/{ruta_csv_1.name}!")
            
        elif sub_op == '2':
            carpeta = obtener_carpeta_guardado("General")
            ruta_csv_2 = carpeta / "12B_Resumen_Medias_Venecia_Acusticas.csv"
            df_resumen_medias.to_csv(ruta_csv_2, index=False, sep=';', decimal=',')
            print(f"✅ ¡CSV de MEDIAS guardado con éxito en: Graficas_Venecia_TFG/General/{ruta_csv_2.name}!")
            
        elif sub_op == '3':
            carpeta = obtener_carpeta_guardado("General")
            ruta_csv_3 = carpeta / "12C_Resumen_Medianas_Venecia_Acusticas.csv"
            df_resumen_medianas.to_csv(ruta_csv_3, index=False, sep=';', decimal=',')
            print(f"✅ ¡CSV de MEDIANAS guardado con éxito en: Graficas_Venecia_TFG/General/{ruta_csv_3.name}!")
            
        elif sub_op == '4':
            localizaciones_disponibles = df_resumen_medias[col_loc].tolist()
            print("\n📍 Localizaciones disponibles:")
            for loc in localizaciones_disponibles:
                print(f"  - {loc}")
                
            while True:
                loc_elegida = input("\n👉 Escribe la plaza para ver sus datos (o '0' para salir al submenú): ").strip()
                
                if loc_elegida == '0':
                    break 
                    
                match = next((l for l in localizaciones_disponibles if l.lower() == loc_elegida.lower()), None)
                if match:
                    loc_media = df_resumen_medias[df_resumen_medias[col_loc] == match].iloc[0]
                    loc_mediana = df_resumen_medianas[df_resumen_medianas[col_loc] == match].iloc[0]
                    
                    print(f"\n🎧 Parámetros Acústicos para: {match} (Encuestas: {loc_media['count']})")
                    print("-" * 50)
                    print(f"{'PARÁMETRO'.ljust(15)} | {'MEDIA'.center(12)} | {'MEDIANA'.center(12)}")
                    print("-" * 50)
                    
                    for col in cols_acusticas_validas:
                        var_name = str(col).upper().ljust(15)
                        val_media = f"{loc_media[col]:.3f}".center(12)
                        val_mediana = f"{loc_mediana[col]:.3f}".center(12)
                        print(f"{var_name} | {val_media} | {val_mediana}")
                    print("-" * 50)
                else:
                    print(f"❌ ERROR: '{loc_elegida}' no existe. Revisa cómo se escribe.")
        
        elif sub_op == '0':
            print("Saliendo de la herramienta de características acústicas...")
            break
        else:
            print("❌ Opción no válida. Por favor, elige 0, 1, 2, 3 o 4.")
def generar_csv_machine_learning_venecia():
    """Opción 13: Genera y guarda el CSV Maestro de Venecia con todos los datos y coordenadas ISO para la IA."""
    print("\n" + "="*60)
    print(" 🤖🎡 GENERANDO CSV MAESTRO PARA MACHINE LEARNING (Venecia) 🎡🤖 ")
    print("="*60)
    
    try:
        # Hacemos una copia profunda del dataframe original limpio (que ya es solo de Venecia)
        df_ml = df_datos.copy()
        
        # Comprobamos si las coordenadas ISO ya están calculadas.
        columnas_actuales = [col.lower() for col in df_ml.columns]
        
        if 'isopleasant' not in columnas_actuales or 'isoeventful' not in columnas_actuales:
            print("⏳ Calculando coordenadas ISOPleasant e ISOEventful...")
            import soundscapy as ssd
            # Calculamos las coordenadas ISO usando la librería nativa
            df_ml = ssd.isd.calculate_paq_coords(df_ml)
        else:
            print("✅ Coordenadas ISO ya detectadas en el dataset de Venecia.")

        # Aseguramos que la carpeta destino exista (asumiendo que tu variable es carpeta_graficas_base)
        carpeta_ml = carpeta_graficas_base / 'General'
        carpeta_ml.mkdir(parents=True, exist_ok=True)
        
        ruta_guardado = carpeta_ml / "ML_Datos_Venecia.csv"
        
        # Guardamos el CSV Maestro
        df_ml.to_csv(ruta_guardado, index=False, sep=';', decimal=',')
        
        print("\n" + "-"*40)
        print(f"✅ ¡ÉXITO! Base de datos de VENECIA lista para la Inteligencia Artificial.")
        print(f"📊 Filas totales (Encuestas válidas en Venecia): {len(df_ml)}")
        print(f"🔢 Columnas totales (Físicas + Subjetivas + ISO): {len(df_ml.columns)}")
        print(f"💾 Guardado en: {ruta_guardado}")
        print("-"*40)
        
    except Exception as e:
        print(f"❌ Error al generar el CSV Maestro para ML: {e}")
# ==========================================
# MENÚ INTERACTIVO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    while True:
        # Aquí le damos el toque visual (pintamos el menú)
        print("\n" + "🇬🇧"*25)
        print(" 🕰️  MENÚ DE ANÁLISIS TFG SOUNDSCAPE (VENECIA) 🎡 ")
        print("🇬🇧"*25)
        print("1. Ver Resumen ISO y Exportar CSV")
        print("2. Generar Modelo circumplejo general (Toda la muestra)")
        print("3. Generar Modelo circumplejo Localizaciones (Individuales o TODAS)")
        print("4. Generar Gráfica ISO por Localización (Densidad y Puntos)")
        print("5. Calcular Parámetros Estadísticos y Gráfica SPI (Realidad vs Objetivo)")
        print("6. Generar Scatter Plot Agrupado por Localizaciones (General)")
        print("7. Generar Gráficos ISO Individuales por Sesión")
        print("8. Generar Scatter Plot de Medias/Medianas ISO General")
        print("9. Generar Joint Plot con Histogramas por Localización")
        print("10. Generar Joint Plot Agrupado (Comparativa a la carta)")
        print("11. Ver Tabla Resumen de Variables Subjetivas (PAQ) y Guardar CSV")
        print("12. Extraer/Consultar Datos Acústicos Objetivos (Medias y Medianas)")
        print("13. 🤖 Generar CSV Maestro para Machine Learning (VENECIA)")
        print("0. Salir")
        
        opcion = input("\nElige una opción (0-13): ").strip()
        
        if opcion == '1':
            resumir_y_exportar_ISO()
        elif opcion == '2':
            generar_radar_circunflejo_paq_global()
        elif opcion == '3':
            generar_radar_localizacion_especifica()
        elif opcion == '4':
            generar_iso_plot_combinado()
        elif opcion == '5':
            calcular_parametros_estatisticos_nativos()
        elif opcion == '6':
            generar_scatter_agrupado_localizacion()
        elif opcion == '7':
            generar_iso_plots_individuales_por_sesion()
        elif opcion == '8':
            generar_scatter_medias_localizaciones()
        elif opcion == '9':
            generar_jointplot_localizacion()
        elif opcion == '10':
            generar_jointplot_agrupado_localizaciones()
        elif opcion == '11':
            resumir_datos_localizaciones()
        elif opcion == '12':
            resumir_caracteristicas_acusticas()
        elif opcion == '13':
            generar_csv_machine_learning_venecia()
        elif opcion == '0':
            print("Saliendo del programa...")
            print("¡Adiós y mucho ánimo con tu análisis de Venecia! 💂‍♂️☕")
            break
        else:
            print("❌ Opción no válida. Por favor, elige un número del 0 al 13.")