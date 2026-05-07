# ==========================================
# Codigo 7 extra: Aprendizaje máquina (Singulares vs Grupos)
# Incorporación de características frecuenciales de audio.
# Modelos: LinearRegression y RandomForest.
# Exportación de CSVs de GridSearch y Rankings añadida.
# ==========================================
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LIBRERÍAS DE EVALUACIÓN Y MÉTRICAS
# ==========================================
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

# ==========================================
# 2. LIBRERÍAS DE MODELOS DE MACHINE LEARNING
# ==========================================
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 3. VALIDACIÓN CRUZADA Y OPTIMIZACIÓN
# ==========================================
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, train_test_split

# ==========================================
# 4. LIBRERÍAS DE VISUALIZACIÓN Y PIPELINE
# ==========================================
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
warnings.filterwarnings('ignore')

# ==========================================
# 5. MENÚ INTERACTIVO Y CARGA DE DATOS
# ==========================================
ruta_script = Path(__file__).parent

def menu_configuracion():
    print("\n" + "="*65)
    print(" 🎛️  MENÚ DE CONFIGURACIÓN DE CARACTERÍSTICAS ACÚSTICAS ")
    print("="*65)
    print("1. SUSTITUIR: Usar SOLO las nuevas características del CSV de audio.")
    print("   (Descartará automáticamente las variables originales como LAeq, LCeq, etc.)")
    print("2. COMBINAR: Usar originales + nuevas características de audio.")
    
    while True:
        opcion = input("\n👉 Elige una opción (1 o 2): ").strip()
        if opcion == '1':
            return '1', ruta_script / 'Graficas_IA_SUSTITUIDO_TFG'
        elif opcion == '2':
            return '2', ruta_script / 'Graficas_IA_COMBINADO_TFG'
        print("❌ Opción inválida. Escribe 1 o 2.")

# Lanzamos el menú y configuramos la carpeta de salida
opcion_elegida, CARPETA_SALIDA = menu_configuracion()
print(f"\n📁 Los resultados de esta ejecución se guardarán en: {CARPETA_SALIDA.name}")

ciudades = ['Londres']
dfs_ciudades = {}

print("\n" + "="*50)
print(" INICIANDO CARGA Y PREPROCESAMIENTO DE DATOS ")
print("="*50)

# --- A. Carga de datos originales (Londres) ---
for ciudad in ciudades:
    nombre_carpeta = f"Graficas_{ciudad}_TFG"
    nombre_archivo = f"ML_Datos_{ciudad}.csv"
    ruta_csv = ruta_script / nombre_carpeta / 'General' / nombre_archivo
    
    if ruta_csv.exists():
        df_temp = pd.read_csv(ruta_csv, sep=';', decimal=',', low_memory=False)
        df_temp.columns = df_temp.columns.str.lower()
        df_temp['origen_ciudad'] = ciudad
        
        # Fundamental pasar las claves a string para que luego el cruce (merge) no falle
        df_temp['recordid'] = df_temp['recordid'].astype(str)
        df_temp['groupid'] = df_temp['groupid'].astype(str)
        df_temp['locationid'] = df_temp['locationid'].astype(str)
        
        if 'uni00' in df_temp.columns:
            df_temp['uni00'] = df_temp['uni00'].astype(str).str.lower().str.strip()
            df_temp['grupo_binario'] = np.where(df_temp['uni00'].isin(['couple', 'group']), 1, 0)
        else:
            df_temp['grupo_binario'] = 0
            
        dfs_ciudades[ciudad] = df_temp
        print(f"✅ {ciudad} cargada correctamente.")
    else:
        print(f"❌ ERROR: No se encuentra el archivo original para {ciudad}.")
        sys.exit()

# --- B. Carga del nuevo CSV de características de audio ---
ruta_audio = ruta_script / 'Graficas_audios_TFG' / 'ISD_Caracteristicas_Frecuenciales.csv'
print(f"\nBuscando características de audio en: {ruta_audio}...")

if ruta_audio.exists():
    df_audio = pd.read_csv(ruta_audio, sep=';', decimal=',', low_memory=False)
    df_audio.columns = df_audio.columns.str.lower()
    
    # Pasamos a string también aquí para asegurar compatibilidad perfecta
    df_audio['recordid'] = df_audio['recordid'].astype(str)
    if 'groupid' in df_audio.columns:
        df_audio['groupid'] = df_audio['groupid'].astype(str)
    if 'locationid' in df_audio.columns:
        df_audio['locationid'] = df_audio['locationid'].astype(str)
    
    # Solo eliminamos 'nombre_audio', conservamos locationid y groupid para el merge
    cols_borrar_audio = ['nombre_audio']
    df_audio = df_audio.drop(columns=[c for c in cols_borrar_audio if c in df_audio.columns])
    
    print(f"✅ Archivo de audio cargado con {len(df_audio.columns) - 3} características frecuenciales.") # -3 por las 3 claves
else:
    print(f"❌ ERROR CRÍTICO: No se encontró el archivo de audio. Verifica la ruta.")
    sys.exit()

# ==============================================================================
# 6. LIMPIEZA, MERGE Y CONSTRUCCIÓN DE MATRICES (X, Y)
# ==============================================================================
print("\n" + "="*50)
print(" PREPARACIÓN DE MATRICES Y CRUCE DE DATOS ")
print("="*50)

# Lista base de columnas que SIEMPRE deben borrarse (metadatos y respuestas que harían trampa)
columnas_prohibidas_fijas = [
    'recordid', 'groupid', 'sessionid', 'locationid', 'start_time', 'end_time', 
    'latitude', 'longitude', 'language', 'survey_version', 'recordinglength',
    'age00','gen00', 'edu00', 'eth00', 'occ00', 'eth00_other', 'occ00_other',
    'misc03', 'misc03_other', 'misc01', 'use00', 'uni00', 'origen_ciudad', 'grupo_binario',
    'ssi01', 'ssi02', 'ssi03', 'ssi04', 'sss01', 'sss02', 'sss03', 'sss04', 'sss05',
    'who01', 'who02', 'who03', 'who04', 'who05', 'who_sum',
    'pleasant', 'vibrant', 'eventful', 'chaotic', 'annoying', 'monotonous', 'uneventful', 'calm',
    'isopleasant', 'isoeventful', 'idp'
]

df_londres_crudo = dfs_ciudades['Londres'].copy()

# Calculamos cuáles son las variables acústicas originales de Londres (Todo lo que no es prohibido)
features_originales = [col for col in df_londres_crudo.columns if col not in columnas_prohibidas_fijas]

# Cruzamos ambas tablas basándonos de manera precisa en 'recordid', 'groupid' y 'locationid'
df_merged = pd.merge(df_londres_crudo, df_audio, on=['recordid', 'groupid', 'locationid'], how='inner')
print(f"🔗 Fusión de alta precisión completada. Registros finales listos para IA: {len(df_merged)}")

# Determinamos qué características descartar según lo que elegiste en el menú
cols_extra_a_borrar = features_originales if opcion_elegida == '1' else []

if opcion_elegida == '1':
    print(f"🛠️ MODO SUSTITUIR: Se eliminarán {len(features_originales)} características originales (LAeq, etc.).")
else:
    print("🛠️ MODO COMBINAR: IA entrenará con el arsenal de características completo (Originales + Audio).")

def aislar_variables_ml(df, columnas_a_descartar):
    if 'isopleasant' not in df.columns or 'isoeventful' not in df.columns:
        raise ValueError("CRÍTICO: Faltan las columnas 'isopleasant' o 'isoeventful'.")
        
    Y_p = df['isopleasant'].fillna(0)
    Y_e = df['isoeventful'].fillna(0)
    mascara_grupos = df['grupo_binario'].copy()
    
    todas_las_prohibidas = columnas_prohibidas_fijas + columnas_a_descartar
    columnas_a_borrar = [col for col in todas_las_prohibidas if col in df.columns]
    
    X = df.drop(columns=columnas_a_borrar).fillna(0)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return X, Y_p, Y_e, mascara_grupos

# Hacemos la partición usando el DataFrame ya fusionado
df_train_crudo, df_test_crudo = train_test_split(
    df_merged, 
    test_size=0.057, 
    random_state=42, 
    stratify=df_merged['locationid']
)

X_train, y_train_p, y_train_e, mask_train_grupos = aislar_variables_ml(df_train_crudo, cols_extra_a_borrar)
X_test,  y_test_p,  y_test_e,  mask_test_grupos  = aislar_variables_ml(df_test_crudo, cols_extra_a_borrar)

print(f"\n📚 SET ENTRENAMIENTO: {len(X_train)} encuestas con {len(X_train.columns)} características (X).")
print(f"📝 SET EXAMEN (Test): {len(X_test)} encuestas.")

# ==============================================================================
# 7. CONSTRUCCIÓN DEL PIPELINE Y GRID SEARCH (OPTIMIZACIÓN DE MODELOS)
# ==============================================================================
print("\n" + "="*50)
print(" INICIANDO PIPELINE Y BÚSQUEDA DE HIPERPARÁMETROS ")
print("="*50)

def optimizar_modelo_gridsearch(X, y, tipo_modelo, target_name):
    print(f"   ⚙️ Configurando Grid Search para: {tipo_modelo}")
    cv_kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    if tipo_modelo == "LinearRegression":
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
        grid_params = {} 
        
    elif tipo_modelo == "RandomForest":
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(random_state=42))])
        grid_params = {
            'model__n_estimators': [5, 10, 15, 20, 25, 40, 50, 100, 200],
            'model__max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'model__min_samples_split': [2, 5, 10]
        }
    else:
        raise ValueError(f"Modelo {tipo_modelo} no soportado.")

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid_params,
        cv=cv_kfold,
        scoring='neg_mean_absolute_error',
        n_jobs=-1, 
        verbose=1 
    )
    
    print(f"   🚀 Iniciando entrenamiento...")
    grid_search.fit(X, y)
    print(f"   ✅ Mejor configuración encontrada: {grid_search.best_params_}")
    
    # GUARDAR RESULTADOS DEL GRIDSEARCH EN CSV
    df_cv_results = pd.DataFrame(grid_search.cv_results_)
    carpeta_gs = CARPETA_SALIDA / 'Resultados_GridSearch'
    carpeta_gs.mkdir(parents=True, exist_ok=True)
    ruta_csv_gs = carpeta_gs / f"GridSearch_{tipo_modelo}_{target_name}.csv"
    df_cv_results.to_csv(ruta_csv_gs, sep=';', index=False, decimal=',')
    print(f"   💾 Detalles del GridSearch exportados a: {ruta_csv_gs.name}")

    return grid_search.best_estimator_

# ==============================================================================
# 8. EJECUCIÓN DEL ENTRENAMIENTO PARA TODOS LOS MODELOS
# ==============================================================================
modelos_a_probar = ["LinearRegression", "RandomForest"]

mejores_modelos_p = {} 
mejores_modelos_e = {} 

print("\n--- ENTRENANDO EXPERTOS EN 'ISOPleasant' ---")
for nombre in modelos_a_probar:
    # Añado la variable 'ISOPleasant' para que el nombre del CSV se guarde correctamente
    mejores_modelos_p[nombre] = optimizar_modelo_gridsearch(X_train, y_train_p, nombre, "ISOPleasant")
    print("-" * 30)

print("\n--- ENTRENANDO EXPERTOS EN 'ISOEventful' ---")
for nombre in modelos_a_probar:
    # Añado la variable 'ISOEventful' para que el nombre del CSV se guarde correctamente
    mejores_modelos_e[nombre] = optimizar_modelo_gridsearch(X_train, y_train_e, nombre, "ISOEventful")
    print("-" * 30)

print("\n🎉 Todos los modelos han sido entrenados y optimizados con éxito.")

# ==============================================================================
# 8.5. EXAMEN FINAL: EVALUACIÓN PUNTUAL DE INDIVIDUOS (SINGLES)
# ==============================================================================
def evaluar_individuos_singles(mejores_modelos, X_test, y_test_real, mask_grupos, target_name):
    print("\n" + "-"*50)
    print(f" EVALUACIÓN DE INDIVIDUOS (Singles) PARA: {target_name.upper()} ")
    print("-"*50)
    
    mask_singles = (mask_grupos == 0)
    X_test_singles = X_test[mask_singles]
    y_test_singles_real = y_test_real[mask_singles]
    
    print(f"   👥 Sujetos aislados encontrados: {len(y_test_singles_real)} personas.")
    
    if len(y_test_singles_real) == 0:
        return pd.DataFrame() 
        
    resultados_evaluacion = []
    
    for nombre_modelo, pipeline_optimo in mejores_modelos.items():
        predicciones = pipeline_optimo.predict(X_test_singles)
        
        mae = mean_absolute_error(y_test_singles_real, predicciones) 
        mse = mean_squared_error(y_test_singles_real, predicciones)  
        rmse = np.sqrt(mse)                                          
        r2 = r2_score(y_test_singles_real, predicciones)             
        
        resultados_evaluacion.append({
            "Modelo": nombre_modelo,
            "R2 (Varianza Explicada)": r2,
            "MAE (Error Absoluto)": mae,
            "RMSE (Error Raíz Cuadrada)": rmse,
            "MSE": mse
        })
        
    df_resultados = pd.DataFrame(resultados_evaluacion)
    df_resultados = df_resultados.sort_values(by="R2 (Varianza Explicada)", ascending=False).reset_index(drop=True)
    df_resultados = df_resultados.round(4)
    
    print(df_resultados.to_string(index=False))

    # GUARDAR CSV DE RANKING SINGLES
    carpeta_resultados = CARPETA_SALIDA / 'Resultados_Metricas'
    carpeta_resultados.mkdir(parents=True, exist_ok=True) 
    df_resultados.to_csv(carpeta_resultados / f"Ranking_Singles_{target_name}.csv", sep=';', index=False, decimal=',')

    return df_resultados

df_singles_p = evaluar_individuos_singles(mejores_modelos_p, X_test, y_test_p, mask_test_grupos, "ISOPleasant")
df_singles_e = evaluar_individuos_singles(mejores_modelos_e, X_test, y_test_e, mask_test_grupos, "ISOEventful")

# ==============================================================================
# 8.6. EXAMEN FINAL: EVALUACIÓN DISTRIBUCIONAL DE GRUPOS/PAREJAS
# ==============================================================================
def calcular_metricas_distribucion(y_true, y_pred):
    if len(y_true) < 2: return 0, 0, 0 
    
    dme = abs(np.mean(y_true) - np.mean(y_pred)) 
    bins = np.linspace(-1, 1, 11)
    
    hist_true, _ = np.histogram(y_true, bins=bins, density=True)
    hist_pred, _ = np.histogram(y_pred, bins=bins, density=True)
    
    p = (hist_true + 1e-10) / np.sum(hist_true + 1e-10)
    q = (hist_pred + 1e-10) / np.sum(hist_pred + 1e-10)
    
    kl = entropy(p, q)            
    js = jensenshannon(p, q)      
    return kl, js, dme

def evaluar_grupos(mejores_modelos, X_test, y_test_real, mask_grupos, target_name):
    print("\n" + "-"*50)
    print(f" EVALUACIÓN DE GRUPOS/PAREJAS PARA: {target_name.upper()} ")
    print("-"*50)
    
    mask_solo_grupos = (mask_grupos == 1)
    X_test_grupos = X_test[mask_solo_grupos]
    y_test_grupos_real = y_test_real[mask_solo_grupos]
    
    print(f"   👥 Sujetos en grupo encontrados: {len(y_test_grupos_real)} personas.")
    
    if len(y_test_grupos_real) < 2: return pd.DataFrame()
        
    resultados_evaluacion = []
    for nombre_modelo, pipeline_optimo in mejores_modelos.items():
        predicciones = pipeline_optimo.predict(X_test_grupos)
        kl, js, dme = calcular_metricas_distribucion(y_test_grupos_real, predicciones)
        
        resultados_evaluacion.append({
            "Modelo": nombre_modelo,
            "JS Divergence (0=Perfecto)": js,
            "KL Divergence": kl,
            "DME (Error Consenso)": dme
        })
        
    df_resultados = pd.DataFrame(resultados_evaluacion)
    df_resultados = df_resultados.sort_values(by="JS Divergence (0=Perfecto)", ascending=True).reset_index(drop=True)
    df_resultados = df_resultados.round(4)
    
    print(df_resultados.to_string(index=False))

    # GUARDAR CSV DE RANKING GRUPOS
    carpeta_resultados = CARPETA_SALIDA / 'Resultados_Metricas'
    carpeta_resultados.mkdir(parents=True, exist_ok=True) 
    df_resultados.to_csv(carpeta_resultados / f"Ranking_Grupos_{target_name}.csv", sep=';', index=False, decimal=',')

    return df_resultados

df_grupos_p = evaluar_grupos(mejores_modelos_p, X_test, y_test_p, mask_test_grupos, "ISOPleasant")
df_grupos_e = evaluar_grupos(mejores_modelos_e, X_test, y_test_e, mask_test_grupos, "ISOEventful")

# ==============================================================================
# 8.7. RESUMEN GLOBAL: UNIENDO MÉTRICAS DE INDIVIDUOS Y GRUPOS
# ==============================================================================
def generar_tabla_resumen_global(df_singles, df_grupos, target_name):
    print("\n" + "="*65)
    print(f" 🌍 TABLA MAESTRA DE RENDIMIENTO PARA: {target_name.upper()} ")
    print("="*65)

    if df_singles.empty or df_grupos.empty: return pd.DataFrame()

    df_global = pd.merge(df_singles, df_grupos, on='Modelo', how='inner')
    df_global = df_global.sort_values(by='MAE (Error Absoluto)', ascending=True).reset_index(drop=True)

    print(df_global.to_string(index=False))
    
    # GUARDAR CSV DEL RANKING GLOBAL (Singles + Grupos unidos)
    carpeta_resultados = CARPETA_SALIDA / 'Resultados_Metricas'
    carpeta_resultados.mkdir(parents=True, exist_ok=True) 
    ruta_csv = carpeta_resultados / f"Resumen_Global_Metricas_{target_name}.csv"
    df_global.to_csv(ruta_csv, sep=';', index=False, decimal=',')
    
    return df_global

df_global_p = generar_tabla_resumen_global(df_singles_p, df_grupos_p, "ISOPleasant")
df_global_e = generar_tabla_resumen_global(df_singles_e, df_grupos_e, "ISOEventful")
# ==============================================================================
# 8.8. AUDITORÍA INDIVIDUAL (TRAIN vs VALIDACIÓN vs TEST)
# ==============================================================================
def generar_auditoria_individual_completa(mejores_modelos, X_train, y_train, X_test, y_test, mask_train, mask_test, target_name):
    print(f"\n" + "⚔️"*25)
    print(f" AUDITORÍA DE FASES DE APRENDIZAJE: {target_name.upper()} ")
    print("⚔️"*25)
    
    carpeta_auditorias = CARPETA_SALIDA / 'Auditorias_Individuales'
    carpeta_auditorias.mkdir(parents=True, exist_ok=True)
    
    mask_singles_train = (mask_train == 0)
    X_train_singles = X_train[mask_singles_train]
    y_train_singles = y_train[mask_singles_train]
    
    mask_singles_test = (mask_test == 0)
    X_test_singles = X_test[mask_singles_test]
    y_test_singles = y_test[mask_singles_test]

    for nombre_modelo, pipeline_optimo in mejores_modelos.items():
        print(f"   📊 Dibujando auditoría de fases para: {nombre_modelo}...")
        
        pred_train = pipeline_optimo.predict(X_train_singles)
        pred_val = cross_val_predict(pipeline_optimo, X_train_singles, y_train_singles, cv=5)
        pred_test = pipeline_optimo.predict(X_test_singles)
        
        filas_plot = []
        fases = [
            ('1. Train\n(Memorización)', y_train_singles, pred_train),
            ('2. Validación\n(CV-5 Pliegues)', y_train_singles, pred_val),
            ('3. Test\n(Groningen)', y_test_singles, pred_test)
        ]
        
        for fase_nombre, y_real, y_pred in fases:
            mse = mean_squared_error(y_real, y_pred)
            filas_plot.extend([
                {'Métrica': 'R2 (Nota /100%)', 'Valor': r2_score(y_real, y_pred), 'Fase': fase_nombre},
                {'Métrica': 'MAE (Error Absoluto)', 'Valor': mean_absolute_error(y_real, y_pred), 'Fase': fase_nombre},
                {'Métrica': 'RMSE (Error Medio)', 'Valor': np.sqrt(mse), 'Fase': fase_nombre},
                {'Métrica': 'MSE (Error² Medio)', 'Valor': mse, 'Fase': fase_nombre}
            ])
            
        df_plot = pd.DataFrame(filas_plot)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Auditoría de Aprendizaje y Generalización\nModelo: {nombre_modelo} ({target_name})', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        metricas_unicas = ['R2 (Nota /100%)', 'MAE (Error Absoluto)', 'RMSE (Error Medio)', 'MSE (Error² Medio)']
        colores_fases = ["#3498db", "#f39c12", "#e74c3c"] 
        
        for i, ax in enumerate(axes.flatten()):
            metrica = metricas_unicas[i]
            df_filtrado = df_plot[df_plot['Métrica'] == metrica]
            
            sns.barplot(data=df_filtrado, x='Fase', y='Valor', ax=ax, palette=colores_fases, edgecolor='black', linewidth=1.5)
            
            hatches = ['///', '...', 'xx']
            for j, bar in enumerate(ax.patches):
                bar.set_hatch(hatches[j % len(hatches)])
                
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', padding=5, fontsize=11, fontweight='bold')
                
            ax.set_title(metrica, fontsize=14, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            
            y_max = df_filtrado['Valor'].max()
            if df_filtrado['Valor'].min() < 0: 
                ax.set_ylim(df_filtrado['Valor'].min() * 1.15, y_max * 1.25)
            else:
                ax.set_ylim(0, y_max * 1.25)
                
        plt.tight_layout(rect=[0, 0, 1, 0.95]) 
        
        nombre_limpio = nombre_modelo.replace('/', '_').replace(' ', '_').replace('.', '')
        ruta_img = carpeta_auditorias / f"Auditoria_Fases_{nombre_limpio}_{target_name}.png"
        plt.savefig(ruta_img, dpi=300, bbox_inches='tight')
        plt.close() 

generar_auditoria_individual_completa(mejores_modelos_p, X_train, y_train_p, X_test, y_test_p, mask_train_grupos, mask_test_grupos, "ISOPleasant")
generar_auditoria_individual_completa(mejores_modelos_e, X_train, y_train_e, X_test, y_test_e, mask_train_grupos, mask_test_grupos, "ISOEventful")
print(f"✅ Auditorías de Fases finalizadas con éxito.")

# ==============================================================================
# 8.9. COMPARATIVA GLOBAL (TODAS LAS IAs AGRUPADAS)
# ==============================================================================
def generar_comparativa_global_ias(df_resumen_global, target_name):
    print(f"\n" + "🌟"*20)
    print(f" DIBUJANDO COMPARATIVA GLOBAL PARA: {target_name.upper()} ")
    print("🌟"*20)

    if df_resumen_global.empty:
        print(f"   ⚠️ No hay datos globales para {target_name}. Saltando gráfica.")
        return

    metricas_cols = ['MAE (Error Absoluto)', 'RMSE (Error Raíz Cuadrada)', 'MSE']
    columnas_presentes = [col for col in metricas_cols if col in df_resumen_global.columns]
    
    if not columnas_presentes:
        print("   ⚠️ Faltan las columnas de métricas necesarias para la comparativa.")
        return

    df_melted = df_resumen_global.melt(
        id_vars='Modelo', 
        value_vars=columnas_presentes, 
        var_name='Métrica', 
        value_name='Valor'
    )
    
    plt.figure(figsize=(15, 8))
    colores = ['#3498db', '#9b59b6', '#2ecc71'] 
    
    ax = sns.barplot(
        data=df_melted, 
        x='Modelo', 
        y='Valor', 
        hue='Métrica', 
        palette=colores, 
        edgecolor='black', 
        linewidth=1.2
    )
    
    plt.title(f'Rendimiento Global de los Algoritmos de IA en Londres\nObjetivo: {target_name.upper()}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=15, ha='right', fontsize=12, fontweight='bold')
    plt.xlabel('Algoritmos de Inteligencia Artificial', fontsize=13, fontweight='bold', labelpad=15)
    plt.ylabel('Puntuación de la Métrica (Más bajo es MEJOR)', fontsize=13, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Métricas de Evaluación', loc='upper right', framealpha=0.9, fontsize=11)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=5, fontsize=10, rotation=90)
        
    y_max = df_melted['Valor'].max()
    y_min = df_melted['Valor'].min()
    plt.ylim(y_min * 1.1 if y_min < 0 else 0, y_max * 1.4)

    plt.tight_layout()
    
    carpeta_comparativas = CARPETA_SALIDA / 'Comparativas_Globales'
    carpeta_comparativas.mkdir(parents=True, exist_ok=True)
    
    ruta_guardado = carpeta_comparativas / f"Comparativa_Global_IAs_{target_name}.png"
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Gráfica comparativa global guardada en: {ruta_guardado.name}")

generar_comparativa_global_ias(df_global_p, "ISOPleasant")
generar_comparativa_global_ias(df_global_e, "ISOEventful")

# ==============================================================================
# 9. EXTRACCIÓN DE INTERPRETABILIDAD (PESOS Y GINI IMPORTANCE)
# ==============================================================================
def extraer_e_imprimir_importancias(mejores_modelos, X_train, y_train, target_name):
    print(f"\n" + "="*50)
    print(f" EXTRAYENDO TOP VARIABLES INDIVIDUALES: {target_name.upper()} ")
    print("="*50)
    
    carpeta_pesos = CARPETA_SALIDA / 'Interpretabilidad'
    carpeta_pesos.mkdir(parents=True, exist_ok=True)
    
    nombres_variables = X_train.columns.tolist()
    
    for nombre_modelo, pipeline_optimo in mejores_modelos.items():
        print(f"   📊 Analizando los secretos de: {nombre_modelo}...")
        importancias = None
        es_coeficiente = False
        
        modelo_real = pipeline_optimo.named_steps['model']
        
        if nombre_modelo == "RandomForest":
            importancias = modelo_real.feature_importances_
        elif nombre_modelo == "LinearRegression":
            importancias = modelo_real.coef_
            es_coeficiente = True
            
        df_imp = pd.DataFrame({'Variable': nombres_variables, 'Valor': importancias})
        
        if es_coeficiente:
            df_imp['Fuerza'] = df_imp['Valor'].abs()
            df_imp = df_imp[df_imp['Fuerza'] > 0.0].sort_values(by='Fuerza', ascending=False).head(15)
        else:
            df_imp = df_imp[df_imp['Valor'] > 0.001].sort_values(by='Valor', ascending=False).head(15)
            
        if len(df_imp) == 0: 
            print(f"      ⚠️ {nombre_modelo} ha anulado todas las variables. Saltando gráfica.")
            continue
            
        plt.figure(figsize=(10, 6))
        
        if es_coeficiente:
            colores = ["#e74c3c" if val < 0 else "#2ecc71" for val in df_imp['Valor']]
            ax = sns.barplot(data=df_imp, x='Valor', y='Variable', palette=colores, edgecolor='black')
            plt.xlabel("Peso del Coeficiente (Impacto direccional)")
            plt.title(f"Top 15 Factores Clave (Coeficientes) - {nombre_modelo}\n({target_name})")
        else:
            ax = sns.barplot(data=df_imp, x='Valor', y='Variable', palette="magma", edgecolor='black')
            plt.xlabel("Nivel de Importancia (Impacto en las decisiones de la IA)")
            plt.title(f"Top 15 Factores Clave (Importancia) - {nombre_modelo}\n({target_name})")
            
        plt.ylabel("Parámetros Acústicos")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        for i, val in enumerate(df_imp['Valor']):
            texto = f"{val:.3f}" if es_coeficiente else f"{val:.4f}" 
            offset = (df_imp['Valor'].max() * 0.02) if val >= 0 else -(abs(df_imp['Valor'].min()) * 0.02)
            ha_align = 'left' if val >= 0 else 'right'
            ax.text(val + offset, i, texto, va='center', ha=ha_align, fontsize=10, fontweight='bold')
            
        x_min, x_max = plt.xlim()
        rango = x_max - x_min
        plt.xlim(x_min - (rango*0.1) if x_min < 0 else 0, x_max + (rango*0.15))
        
        plt.tight_layout()
        
        ruta_img = carpeta_pesos / f"Pesos_{nombre_modelo.replace('.', '')}_{target_name}.png"
        plt.savefig(ruta_img, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✅ Gráfica generada: {ruta_img.name}")

extraer_e_imprimir_importancias(mejores_modelos_p, X_train, y_train_p, "ISOPleasant")
extraer_e_imprimir_importancias(mejores_modelos_e, X_train, y_train_e, "ISOEventful")

# ==============================================================================
# 10. ENSAMBLAJE (ENVERGUDURA GLOBAL): LA "SÚPER PREDICCIÓN"
# ==============================================================================
def generar_super_prediccion_ensamblada(mejores_modelos, df_rank_singles, X_test, y_test_real, target_name):
    print("\n" + "🌟"*25)
    print(f" GENERANDO SÚPER PREDICCIÓN ENSAMBLADA: {target_name.upper()} ")
    print("🌟"*25)
    
    if df_rank_singles.empty: return
        
    lineales = ["LinearRegression"]
    complejos = ["RandomForest"]
    
    df_lineales = df_rank_singles[df_rank_singles['Modelo'].isin(lineales)]
    df_complejos = df_rank_singles[df_rank_singles['Modelo'].isin(complejos)]
    
    if df_lineales.empty or df_complejos.empty: return
        
    mejor_lineal = df_lineales.sort_values(by='MAE (Error Absoluto)').iloc[0]['Modelo']
    mejor_complejo = df_complejos.sort_values(by='MAE (Error Absoluto)').iloc[0]['Modelo']
    
    print(f"   🏆 Campeón Lineal: {mejor_lineal}")
    print(f"   🏆 Campeón Complejo: {mejor_complejo}")
    
    prediccion_lineal = mejores_modelos[mejor_lineal].predict(X_test)
    prediccion_compleja = mejores_modelos[mejor_complejo].predict(X_test)
    
    super_prediccion = (prediccion_lineal + prediccion_compleja) / 2.0
    
    mae_super = mean_absolute_error(y_test_real, super_prediccion)
    rmse_super = np.sqrt(mean_squared_error(y_test_real, super_prediccion))
    r2_super = r2_score(y_test_real, super_prediccion)
    
    print(f"   📊 SÚPER MODELO -> R2: {r2_super:.4f} | MAE: {mae_super:.4f} | RMSE: {rmse_super:.4f}")
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test_real, y=super_prediccion, alpha=0.6, color='#8e44ad', s=60, edgecolor='white')
    
    limite_min = min(y_test_real.min(), super_prediccion.min()) - 0.1
    limite_max = max(y_test_real.max(), super_prediccion.max()) + 0.1
    plt.plot([limite_min, limite_max], [limite_min, limite_max], color='red', linestyle='--', linewidth=2)
    
    plt.title(f'Realidad vs Súper Predicción Ensamblada\n({mejor_lineal} + {mejor_complejo}) - {target_name}', fontsize=14, pad=15)
    plt.xlabel('Respuesta Real (Encuestas)')
    plt.ylabel('Súper Predicción (IA Ensamblada)')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    texto_box = f"$R^2$ = {r2_super:.4f}\nMAE = {mae_super:.4f}"
    plt.text(0.05, 0.95, texto_box, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
             
    plt.tight_layout()
    carpeta_ensamble = CARPETA_SALIDA / 'Ensamblaje'
    carpeta_ensamble.mkdir(parents=True, exist_ok=True)
    plt.savefig(carpeta_ensamble / f"Super_Prediccion_{target_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

generar_super_prediccion_ensamblada(mejores_modelos_p, df_singles_p, X_test, y_test_p, "ISOPleasant")
generar_super_prediccion_ensamblada(mejores_modelos_e, df_singles_e, X_test, y_test_e, "ISOEventful")

print("\n" + "🚀"*20)
print(" PIPELINE DE MACHINE LEARNING FINALIZADO COMPLETAMENTE ")
print("🚀"*20)