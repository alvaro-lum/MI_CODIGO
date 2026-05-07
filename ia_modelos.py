import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LIBRERÍAS DE EVALUACIÓN Y MÉTRICAS
# ==========================================
# Las típicas para medir cuánto se equivoca el modelo (MAE, RMSE) y cuánto acierta (R2)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Estas dos son para medir si la distribución de las respuestas de un grupo 
# se parece a lo que predice la IA
from scipy.stats import entropy 
from scipy.spatial.distance import jensenshannon 

# ==========================================
# 2. LIBRERÍAS DE MODELOS DE MACHINE LEARNING
# ==========================================
# Batería reducida según requerimientos: un modelo lineal simple y un ensamble de árboles
from sklearn.linear_model import LinearRegression      # Modelo lineal básico
from sklearn.ensemble import RandomForestRegressor     # Basado en múltiples árboles de decisión

# ==========================================
# 3. VALIDACIÓN CRUZADA Y OPTIMIZACIÓN (GRID SEARCH)
# ==========================================
# Herramientas para probar distintas configuraciones de cada modelo y quedarme con la mejor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict

# ==========================================
# 4. LIBRERÍAS DE VISUALIZACIÓN Y PIPELINE
# ==========================================
# Pipeline es para encadenar pasos (ej: primero escalar datos, luego entrenar) de forma limpia
from sklearn.pipeline import Pipeline
# StandardScaler sirve para que todas las variables estén en la misma escala
from sklearn.preprocessing import StandardScaler
# Oculto los "warnings" (avisos en rojo) de la terminal
warnings.filterwarnings('ignore')

# ==========================================
# 5. CONFIGURACIÓN DE RUTAS Y CARGA DE DATOS
# ==========================================
# Saco la ruta exacta donde está este script guardado para no tener problemas con rutas absolutas
ruta_script = Path(__file__).parent
ciudades = ['Londres', 'Venecia', 'Granada', 'Groningen']

# Aquí voy a guardar los DataFrames de cada ciudad temporalmente
dfs_ciudades = {}

print("\n" + "="*50)
print(" INICIANDO CARGA Y PREPROCESAMIENTO DE DATOS ")
print("="*50)

# Bucle para leer los CSVs de cada ciudad
for ciudad in ciudades:
    # Construyo la ruta del archivo dinámicamente
    nombre_carpeta = f"Graficas_{ciudad}_TFG"
    nombre_archivo = f"ML_Datos_{ciudad}.csv"
    ruta_csv = ruta_script / nombre_carpeta / 'General' / nombre_archivo
    
    if ruta_csv.exists():
        # Leo el CSV. Uso sep=';' y decimal=',' porque los datos vienen con formato europeo
        df_temp = pd.read_csv(ruta_csv, sep=';', decimal=',', low_memory=False)
        # Paso los nombres de las columnas a minúsculas
        df_temp.columns = df_temp.columns.str.lower()
        # Añado una columna para saber de qué ciudad viene cada fila
        df_temp['origen_ciudad'] = ciudad
        
        # --- PREPROCESAMIENTO DE 'UNI00' (INDIVIDUOS VS GRUPOS) ---
        if 'uni00' in df_temp.columns:
            # Limpio el texto
            df_temp['uni00'] = df_temp['uni00'].astype(str).str.lower().str.strip()
            # Creo una máscara: 1 si estaban en pareja/grupo, 0 si estaban solos (Single)
            df_temp['grupo_binario'] = np.where(df_temp['uni00'].isin(['couple', 'group']), 1, 0)
        else:
            print(f"   ⚠️ ADVERTENCIA: La columna 'uni00' no existe en {ciudad}. Asumiendo todos como 'Single' (0).")
            df_temp['grupo_binario'] = 0
            
        dfs_ciudades[ciudad] = df_temp
        print(f"✅ {ciudad} cargada: {len(df_temp)} encuestas. (Individuos: {(df_temp['grupo_binario']==0).sum()} | Grupos: {(df_temp['grupo_binario']==1).sum()})")
    else:
        print(f"❌ ERROR: No se encuentra el archivo para {ciudad}.")

print("\nCarga finalizada. Datos listos en memoria.")

# ==============================================================================
# 6. LIMPIEZA, SEPARACIÓN ESPACIAL Y CONSTRUCCIÓN DE MATRICES MATEMÁTICAS (X, Y)
# ==============================================================================
print("\n" + "="*50)
print(" PREPARACIÓN DE MATRICES (TRAIN vs TEST ESPACIAL) ")
print("="*50)

def aislar_variables_ml(df):
    """
    Función para separar lo que quiero predecir (Y) de las variables que uso para predecir (X).
    También elimino columnas que harían "trampa" o que no aportan nada al algoritmo.
    """
    if 'isopleasant' not in df.columns or 'isoeventful' not in df.columns:
        raise ValueError("CRÍTICO: Faltan las columnas 'isopleasant' o 'isoeventful' en el CSV.")
        
    # Mis dos grandes objetivos a predecir (Agudo/Placentero y Eventful/Animado)
    Y_p = df['isopleasant'].fillna(0)
    Y_e = df['isoeventful'].fillna(0)
    # Guardo si es grupo o no para evaluar después por separado
    mascara_grupos = df['grupo_binario'].copy()
    
    # Lista de columnas basura o que son variables objetivo directas
    columnas_prohibidas = [
        'recordid', 'groupid', 'sessionid', 'locationid', 'start_time', 'end_time', 
        'latitude', 'longitude', 'language', 'survey_version', 'recordinglength',
        'age00','gen00', 'edu00', 'eth00', 'occ00', 'eth00_other', 'occ00_other',
        'misc03', 'misc03_other', 'misc01', 'use00', 'uni00', 'origen_ciudad', 'grupo_binario',
        'ssi01', 'ssi02', 'ssi03', 'ssi04', 'sss01', 'sss02', 'sss03', 'sss04', 'sss05',
        'who01', 'who02', 'who03', 'who04', 'who05', 'who_sum',
        'pleasant', 'vibrant', 'eventful', 'chaotic', 'annoying', 'monotonous', 'uneventful', 'calm',
        'isopleasant', 'isoeventful', 'idp'
    ]
    
    # Borro las columnas prohibidas que existan realmente en mi DataFrame
    columnas_a_borrar = [col for col in columnas_prohibidas if col in df.columns]
    X = df.drop(columns=columnas_a_borrar).fillna(0)
    # Me aseguro de que todo lo que queda en la matriz X sea numérico
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return X, Y_p, Y_e, mascara_grupos

# === ESTRATEGIA DE VALIDACIÓN ESPACIAL ===
# Entreno con 3 ciudades y examino al modelo con una ciudad que NUNCA ha visto.
ciudad_test = 'Groningen'

if ciudad_test in dfs_ciudades:
    print(f"🎯 Variables objetivo configuradas: ISOPLEASANT e ISOEVENTFUL")
    
    # El examen final (datos nunca vistos)
    df_test_crudo = dfs_ciudades[ciudad_test].copy()
    
    # Junto los datos de las otras 3 ciudades para crear el material de estudio de la IA
    lista_entrenamiento = [df for ciudad, df in dfs_ciudades.items() if ciudad != ciudad_test]
    df_train_crudo = pd.concat(lista_entrenamiento, ignore_index=True)
    
    # Aplico mi función de limpieza para separar las X (preguntas) de las Y (respuestas)
    X_train, y_train_p, y_train_e, mask_train_grupos = aislar_variables_ml(df_train_crudo)
    X_test,  y_test_p,  y_test_e,  mask_test_grupos  = aislar_variables_ml(df_test_crudo)
    
    print(f"📚 SET DE ENTRENAMIENTO (Train): {len(X_train)} encuestas (Londres, Venecia, Granada).")
    print(f"   -> Variables predictoras (Columnas X resultantes): {len(X_train.columns)}")
    print(f"📝 SET DE EXAMEN (Test Secreto): {len(X_test)} encuestas ({ciudad_test}).")
else:
    raise ValueError(f"CRÍTICO: No se han cargado los datos de {ciudad_test} para usar como Test.")

def optimizar_modelo_gridsearch(X, y, tipo_modelo, target_name):
    """
    Esta función entrena un modelo concreto probando combinaciones de hiperparámetros 
    para encontrar la configuración que comete menos errores. Ahora también guarda
    un CSV con el historial de todas las pruebas realizadas.
    """
    print(f"   ⚙️ Configurando Grid Search para: {tipo_modelo}")
    cv_kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    if tipo_modelo == "LinearRegression":
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
        grid_params = {} 
        
    elif tipo_modelo == "RandomForest":
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(random_state=42))])
        grid_params = {
            'model__n_estimators': [5, 10, 15, 20, 25, 40, 50, 100, 200],    #5, 10, 15, 20, 25, 40, 50, 100, 200
            'model__max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],       #1, 2, 3, 4 5, 6, 7, 8, 9,10, 11, 12, 13, 14, 15, 16, 17 ,18, 19, 20
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
    
    print(f"   🚀 Iniciando entrenamiento (esto puede tardar unos minutos)...")
    grid_search.fit(X, y)
    print(f"   ✅ Mejor configuración encontrada: {grid_search.best_params_}")
    
    # =================================================================
    # NUEVO: GUARDAR EL HISTORIAL DEL GRID SEARCH EN UN CSV
    # =================================================================
    # grid_search.cv_results_ contiene un diccionario con todo lo que ha probado
    df_cv_results = pd.DataFrame(grid_search.cv_results_)
    
    # Ordenamos la tabla para que la configuración ganadora (rank 1) salga arriba
    df_cv_results = df_cv_results.sort_values(by='rank_test_score').reset_index(drop=True)
    
    # Creamos una carpeta específica para estos historiales
    carpeta_gs = ruta_script / 'Prueba_TFG' / 'Resultados_GridSearch'
    carpeta_gs.mkdir(parents=True, exist_ok=True)
    
    # Guardamos el CSV (usando separador europeo)
    ruta_csv_gs = carpeta_gs / f"GridSearch_Historial_{tipo_modelo}_{target_name}.csv"
    df_cv_results.to_csv(ruta_csv_gs, sep=';', index=False, decimal=',')
    print(f"   💾 Historial de hiperparámetros guardado en: {ruta_csv_gs.name}")
    # =================================================================
    
    return grid_search.best_estimator_
# ==============================================================================
# 8. EJECUCIÓN DEL ENTRENAMIENTO PARA TODOS LOS MODELOS
# ==============================================================================
modelos_a_probar = ["LinearRegression", "RandomForest"]

# Diccionarios donde guardaré las IAs ya entrenadas y listas para usar
mejores_modelos_p = {} # Modelos que predicen ISOPleasant
mejores_modelos_e = {} # Modelos que predicen ISOEventful

print("\n--- ENTRENANDO EXPERTOS EN 'ISOPleasant' ---")
for nombre in modelos_a_probar:
    # Le pasamos "ISOPleasant" al final
    mejores_modelos_p[nombre] = optimizar_modelo_gridsearch(X_train, y_train_p, nombre, "ISOPleasant")
    print("-" * 30)

print("\n--- ENTRENANDO EXPERTOS EN 'ISOEventful' ---")
for nombre in modelos_a_probar:
    # Le pasamos "ISOEventful" al final
    mejores_modelos_e[nombre] = optimizar_modelo_gridsearch(X_train, y_train_e, nombre, "ISOEventful")
    print("-" * 30)

print("\n🎉 Todos los modelos han sido entrenados y optimizados con éxito.")

# ==============================================================================
# 8.5. EXAMEN FINAL: EVALUACIÓN PUNTUAL DE INDIVIDUOS (SINGLES)
# ==============================================================================
def evaluar_individuos_singles(mejores_modelos, X_test, y_test_real, mask_grupos, target_name):
    """
    Filtro SÓLO a las personas que rellenaron solas (Singles), y calculo métricas.
    """
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
        # Predicción
        predicciones = pipeline_optimo.predict(X_test_singles)
        
        # Comparación vs realidad
        mae = mean_absolute_error(y_test_singles_real, predicciones) 
        mse = mean_squared_error(y_test_singles_real, predicciones)  
        rmse = np.sqrt(mse)                                          
        r2 = r2_score(y_test_singles_real, predicciones)             
        
        resultados_evaluacion.append({
            "Modelo": nombre_modelo,
            "R2 (Varianza Explicada)": r2,
            "MAE (Error Absoluto)": mae,
            "RMSE (Error Raíz Cuadrada)": rmse,
            "MSE (Error² Medio)": mse
        })
        
    df_resultados = pd.DataFrame(resultados_evaluacion)
    df_resultados = df_resultados.sort_values(by="R2 (Varianza Explicada)", ascending=False).reset_index(drop=True)
    df_resultados = df_resultados.round(4)
    
    print(df_resultados.to_string(index=False))
    return df_resultados

df_singles_p = evaluar_individuos_singles(mejores_modelos_p, X_test, y_test_p, mask_test_grupos, "ISOPleasant")
df_singles_e = evaluar_individuos_singles(mejores_modelos_e, X_test, y_test_e, mask_test_grupos, "ISOEventful")

# ==============================================================================
# 8.6. EXAMEN FINAL: EVALUACIÓN DISTRIBUCIONAL DE GRUPOS/PAREJAS
# ==============================================================================
def calcular_metricas_distribucion(y_true, y_pred):
    """
    Compara si "la opinión general del grupo" y "la distribución predicha" se parecen.
    """
    if len(y_true) < 2: return 0, 0, 0 
    
    dme = abs(np.mean(y_true) - np.mean(y_pred)) # Diferencia de medias simple
    
    bins = np.linspace(-1, 1, 11)
    
    hist_true, _ = np.histogram(y_true, bins=bins, density=True)
    hist_pred, _ = np.histogram(y_pred, bins=bins, density=True)
    
    p = (hist_true + 1e-10) / np.sum(hist_true + 1e-10)
    q = (hist_pred + 1e-10) / np.sum(hist_pred + 1e-10)
    
    kl = entropy(p, q)            # Divergencia KL
    js = jensenshannon(p, q)      # Divergencia JS
    return kl, js, dme

def evaluar_grupos(mejores_modelos, X_test, y_test_real, mask_grupos, target_name):
    """
    Filtra solo a los que respondieron en grupo y les aplica las métricas de distribución.
    """
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
    return df_resultados

df_grupos_p = evaluar_grupos(mejores_modelos_p, X_test, y_test_p, mask_test_grupos, "ISOPleasant")
df_grupos_e = evaluar_grupos(mejores_modelos_e, X_test, y_test_e, mask_test_grupos, "ISOEventful")

# ==============================================================================
# 8.7. RESUMEN GLOBAL: UNIENDO MÉTRICAS DE INDIVIDUOS Y GRUPOS
# ==============================================================================
def generar_tabla_resumen_global(df_singles, df_grupos, target_name):
    """
    Junta la tabla de resultados de individuos con la de grupos por el nombre del modelo
    y guarda el resultado en un CSV.
    """
    print("\n" + "="*65)
    print(f" 🌍 TABLA MAESTRA DE RENDIMIENTO PARA: {target_name.upper()} ")
    print("="*65)

    if df_singles.empty or df_grupos.empty: return pd.DataFrame()

    df_global = pd.merge(df_singles, df_grupos, on='Modelo', how='inner')
    df_global = df_global.sort_values(by='MAE (Error Absoluto)', ascending=True).reset_index(drop=True)

    print(df_global.to_string(index=False))
    
    carpeta_resultados = ruta_script / 'Prueba_TFG' / 'Resultados_Metricas'
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
    """
    Genera una gráfica 2x2 para cada modelo, evaluando su comportamiento en 3 fases:
    1. Entrenamiento
    2. Validación Cruzada
    3. Test (Groningen)
    Filtramos solo a los individuos solitarios (Singles).
    """
    print(f"\n" + "⚔️"*25)
    print(f" AUDITORÍA DE FASES DE APRENDIZAJE: {target_name.upper()} ")
    print("⚔️"*25)
    
    carpeta_auditorias = ruta_script / 'Prueba_TFG' / 'Auditorias_Individuales'
    carpeta_auditorias.mkdir(parents=True, exist_ok=True)
    
    mask_singles_train = (mask_train == 0)
    X_train_singles = X_train[mask_singles_train]
    y_train_singles = y_train[mask_singles_train]
    
    mask_singles_test = (mask_test == 0)
    X_test_singles = X_test[mask_singles_test]
    y_test_singles = y_test[mask_singles_test]

    for nombre_modelo, pipeline_optimo in mejores_modelos.items():
        print(f"   📊 Dibujando auditoría de fases para: {nombre_modelo}...")
        
        # Fase 1: Entrenamiento puro
        pred_train = pipeline_optimo.predict(X_train_singles)
        
        # Fase 2: Validación cruzada (cv=5)
        pred_val = cross_val_predict(pipeline_optimo, X_train_singles, y_train_singles, cv=5)
        
        # Fase 3: Test (Groningen)
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
        
        # DIBUJAR LA GRÁFICA 2x2
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
    """
    Genera una gráfica de barras comparando MAE, RMSE y MSE para los modelos seleccionados.
    """
    print(f"\n" + "🌟"*20)
    print(f" DIBUJANDO COMPARATIVA GLOBAL PARA: {target_name.upper()} ")
    print("🌟"*20)

    if df_resumen_global.empty:
        print(f"   ⚠️ No hay datos globales para {target_name}. Saltando gráfica.")
        return

    metricas_cols = ['MAE (Error Absoluto)', 'RMSE (Error Raíz Cuadrada)', 'MSE (Error² Medio)']
    columnas_presentes = [col for col in metricas_cols if col in df_resumen_global.columns]
    
    if len(columnas_presentes) == 0:
        print("   ❌ ERROR: No se encontraron las métricas de error en el DataFrame.")
        return

    df_melted = df_resumen_global.melt(
        id_vars='Modelo', 
        value_vars=columnas_presentes, 
        var_name='Métrica', 
        value_name='Valor'
    )
    
    plt.figure(figsize=(10, 8)) # Tamaño ligeramente ajustado al tener menos modelos
    colores = ['#3498db', '#f39c12', '#2ecc71'] 
    
    ax = sns.barplot(
        data=df_melted, 
        x='Modelo', 
        y='Valor', 
        hue='Métrica', 
        palette=colores, 
        edgecolor='black', 
        linewidth=1.2
    )
    
    plt.title(f'Rendimiento Global de los Algoritmos de IA en Groningen\nObjetivo: {target_name.upper()}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=0, fontsize=12, fontweight='bold') # Sin rotación porque son solo 2 nombres
    plt.xlabel('Algoritmos de Inteligencia Artificial', fontsize=13, fontweight='bold', labelpad=15)
    plt.ylabel('Puntuación de la Métrica (Más bajo es MEJOR)', fontsize=13, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.legend(title='Métricas de Evaluación', loc='upper right', framealpha=0.9, fontsize=11)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=5, fontsize=10) # Rotación quitada para mejor lectura
        
    y_max = df_melted['Valor'].max()
    plt.ylim(0, y_max * 1.4)

    plt.tight_layout()
    
    carpeta_comparativas = ruta_script / 'Prueba_TFG' / 'Comparativas_Globales'
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
    """
    Entra en "el cerebro" de cada IA y extrae qué características del sonido pesaron más.
    - Regresión Lineal: Nos da Coeficientes (+/- influye).
    - Random Forest: Nos da Gini Importance (% de uso del dato en los árboles).
    """
    print(f"\n" + "="*50)
    print(f" EXTRAYENDO TOP VARIABLES INDIVIDUALES: {target_name.upper()} ")
    print("="*50)
    
    carpeta_pesos = ruta_script / 'Prueba_TFG' / 'Interpretabilidad'
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
            
        # --- PROCESADO DE DATOS PARA LA GRÁFICA ---
        df_imp = pd.DataFrame({'Variable': nombres_variables, 'Valor': importancias})
        
        if es_coeficiente:
            df_imp['Fuerza'] = df_imp['Valor'].abs()
            df_imp = df_imp[df_imp['Fuerza'] > 0.0].sort_values(by='Fuerza', ascending=False).head(15)
        else:
            df_imp = df_imp[df_imp['Valor'] > 0.001].sort_values(by='Valor', ascending=False).head(15)
            
        if len(df_imp) == 0: 
            print(f"      ⚠️ {nombre_modelo} ha anulado todas las variables. Saltando gráfica.")
            continue
            
        # --- DIBUJO DE LA GRÁFICA ---
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
    """
    Construyo un "Frankenstein" con los dos modelos: Linear Regression y Random Forest.
    """
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
    carpeta_ensamble = ruta_script / 'Prueba_TFG' / 'Ensamblaje'
    carpeta_ensamble.mkdir(parents=True, exist_ok=True)
    plt.savefig(carpeta_ensamble / f"Super_Prediccion_{target_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

generar_super_prediccion_ensamblada(mejores_modelos_p, df_singles_p, X_test, y_test_p, "ISOPleasant")
generar_super_prediccion_ensamblada(mejores_modelos_e, df_singles_e, X_test, y_test_e, "ISOEventful")

print("\n" + "🚀"*20)
print(" PIPELINE DE MACHINE LEARNING FINALIZADO COMPLETAMENTE ")
print("🚀"*20)