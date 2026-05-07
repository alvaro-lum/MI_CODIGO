# ==========================================
# Codigo 1: aprendizaje máquina con la particularidad de separar los datos en singulares de grupos/parejas.
# Train --> Londres, Granda, Venecia. Test --> Groningen.
# ==========================================
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
# se parece a lo que predice la IA (útil para cuando evalúo grupos/parejas en lugar de individuos)
from scipy.stats import entropy # Para calcular KL Divergence (Kullback-Leibler)
from scipy.spatial.distance import jensenshannon # Para calcular JS Divergence (Jensen-Shannon)

# ==========================================
# 2. LIBRERÍAS DE MODELOS DE MACHINE LEARNING
# ==========================================
# Aquí importo mi "batería" de inteligencias artificiales, de más simples a más complejas:
from sklearn.linear_model import LinearRegression, Lasso # Modelos lineales básicos
from sklearn.ensemble import RandomForestRegressor       # Basado en múltiples árboles de decisión
from sklearn.svm import SVR                              # Support Vector Regression (busca márgenes de error)
from sklearn.gaussian_process import GaussianProcessRegressor # Basado en probabilidad Bayesiana
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C # Funciones matemáticas para el Gaussian Process
import xgboost as xgb                                    # El rey de los árboles potenciados (Gradient Boosting)

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
# StandardScaler sirve para que todas las variables estén en la misma escala (ej: que un valor de 1000 no pese más que uno de 1)
from sklearn.preprocessing import StandardScaler
# Gini importance
from sklearn.inspection import permutation_importance
# Oculto los "warnings" (avisos en rojo) de la terminal para que no me ensucie los prints de resultados
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
        # Paso los nombres de las columnas a minúsculas para evitar errores tipográficos luego
        df_temp.columns = df_temp.columns.str.lower()
        # Añado una columna para saber de qué ciudad viene cada fila (útil al juntarlas)
        df_temp['origen_ciudad'] = ciudad
        
        # --- PREPROCESAMIENTO DE 'UNI00' (INDIVIDUOS VS GRUPOS) ---
        # Me interesa saber si la encuesta la rellenó alguien solo o acompañado
        if 'uni00' in df_temp.columns:
            # Limpio el texto (todo a minúsculas y sin espacios a los lados)
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
    
    # Lista de columnas basura o que son variables objetivo directas (si las dejo, el modelo haría trampa)
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
# Para demostrar que mi IA generaliza bien y no solo "memoriza",
# entreno con 3 ciudades y examino al modelo con una ciudad que NUNCA ha visto.
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

# ==============================================================================
# 7. CONSTRUCCIÓN DEL PIPELINE Y GRID SEARCH (OPTIMIZACIÓN DE MODELOS)
# ==============================================================================
print("\n" + "="*50)
print(" INICIANDO PIPELINE Y BÚSQUEDA DE HIPERPARÁMETROS ")
print("="*50)

def optimizar_modelo_gridsearch(X, y, tipo_modelo):
    """
    Esta función entrena un modelo concreto, pero no lo hace de cualquier manera.
    Prueba muchas combinaciones de "ruedas y tuercas" (hiperparámetros) para encontrar 
    la configuración que comete menos errores (usando validación cruzada).
    """
    print(f"   ⚙️ Configurando Grid Search para: {tipo_modelo}")
    # KFold(5) significa que divide el Train en 5 trozos: entrena con 4 y se evalúa con 1, rotando.
    cv_kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Defino el Pipeline (primero escalo, luego el modelo) y los parámetros a probar según el modelo
    if tipo_modelo == "LinearRegression":
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
        grid_params = {} # La regresión lineal clásica no tiene tuercas que ajustar
        
    elif tipo_modelo == "Lasso":
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', Lasso(random_state=42))])
        grid_params = {'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]} # Probando distintos niveles de penalización
        
    elif tipo_modelo == "RandomForest":
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(random_state=42))])
        grid_params = {
            'model__n_estimators': [50, 100, 200],    # Cuántos árboles planto
            'model__max_depth': [None, 10, 20],       # Hasta qué profundidad crecen
            'model__min_samples_split': [2, 5, 10]    # Cuándo dejo de ramificar
        }
        
    elif tipo_modelo == "XGBoost":
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', xgb.XGBRegressor(random_state=42, objective='reg:squarederror'))])
        grid_params = {
            'model__learning_rate': [0.01, 0.1, 0.2], # Velocidad de aprendizaje
            'model__max_depth': [3, 5, 7],
            'model__subsample': [0.8, 1.0]
        }
        
    elif tipo_modelo == "SVR":
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', SVR())])
        grid_params = {
            'model__kernel': ['rbf', 'linear'],       # Cómo transformo el espacio de datos
            'model__C': [0.1, 1.0, 10.0],             # Flexibilidad de los márgenes
            'model__epsilon': [0.01, 0.1, 0.5]
        }
        
    elif tipo_modelo == "GaussianProcess":
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', GaussianProcessRegressor(random_state=42))])
        grid_params = {
            'model__kernel': [1.0 * RBF(1.0), C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))],
            'model__alpha': [1e-2, 1e-1, 1.0]         # Nivel de ruido asumido
        }
    else:
        raise ValueError(f"Modelo {tipo_modelo} no soportado.")

    # Monto el "motor de búsqueda". Le digo que me optimice el MAE (Error Absoluto Medio).
    # Como Scikit-learn siempre busca "maximizar" métricas, usa la versión negativa del MAE.
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid_params,
        cv=cv_kfold,
        scoring='neg_mean_absolute_error',
        n_jobs=-1, # Uso todos los núcleos de mi procesador para ir más rápido
        verbose=1 
    )
    
    print(f"   🚀 Iniciando entrenamiento (esto puede tardar unos minutos)...")
    grid_search.fit(X, y)
    print(f"   ✅ Mejor configuración encontrada: {grid_search.best_params_}")
    
    # Devuelvo solo la versión del modelo que ganó la competición interna
    return grid_search.best_estimator_

# ==============================================================================
# 8. EJECUCIÓN DEL ENTRENAMIENTO PARA TODOS LOS MODELOS
# ==============================================================================
modelos_a_probar = [
    "LinearRegression", "Lasso", "RandomForest", 
    "XGBoost", "SVR", "GaussianProcess"
]

# Diccionarios donde guardaré las IAs ya entrenadas y listas para usar
mejores_modelos_p = {} # Modelos que predicen ISOPleasant
mejores_modelos_e = {} # Modelos que predicen ISOEventful

print("\n--- ENTRENANDO EXPERTOS EN 'ISOPleasant' ---")
for nombre in modelos_a_probar:
    mejores_modelos_p[nombre] = optimizar_modelo_gridsearch(X_train, y_train_p, nombre)
    print("-" * 30)

print("\n--- ENTRENANDO EXPERTOS EN 'ISOEventful' ---")
for nombre in modelos_a_probar:
    mejores_modelos_e[nombre] = optimizar_modelo_gridsearch(X_train, y_train_e, nombre)
    print("-" * 30)

print("\n🎉 Todos los modelos han sido entrenados y optimizados con éxito.")

# ==============================================================================
# 8.5. EXAMEN FINAL: EVALUACIÓN PUNTUAL DE INDIVIDUOS (SINGLES)
# ==============================================================================
def evaluar_individuos_singles(mejores_modelos, X_test, y_test_real, mask_grupos, target_name):
    """
    Aquí cojo la ciudad secreta (Groningen), filtro SÓLO a las personas que rellenaron
    la encuesta solas (Singles), y calculo las métricas de predicción clásicas.
    """
    print("\n" + "-"*50)
    print(f" EVALUACIÓN DE INDIVIDUOS (Singles) PARA: {target_name.upper()} ")
    print("-"*50)
    
    mask_singles = (mask_grupos == 0)
    X_test_singles = X_test[mask_singles]
    y_test_singles_real = y_test_real[mask_singles]
    
    print(f"   👥 Sujetos aislados encontrados: {len(y_test_singles_real)} personas.")
    
    if len(y_test_singles_real) == 0:
        return pd.DataFrame() # Salida de emergencia por si no hay datos
        
    resultados_evaluacion = []
    
    for nombre_modelo, pipeline_optimo in mejores_modelos.items():
        # Que el modelo haga su predicción
        predicciones = pipeline_optimo.predict(X_test_singles)
        
        # Calculo lo que se ha equivocado comparando predicción vs realidad
        mae = mean_absolute_error(y_test_singles_real, predicciones) # Error promedio directo
        mse = mean_squared_error(y_test_singles_real, predicciones)  # Penaliza errores gordos
        rmse = np.sqrt(mse)                                          # Devuelve el MSE a la escala original
        r2 = r2_score(y_test_singles_real, predicciones)             # % de la varianza que he logrado explicar
        
        resultados_evaluacion.append({
            "Modelo": nombre_modelo,
            "R2 (Varianza Explicada)": r2,
            "MAE (Error Absoluto)": mae,
            "RMSE (Error Raíz Cuadrada)": rmse,
            "MSE (Error² Medio)": mse
        })
        
    df_resultados = pd.DataFrame(resultados_evaluacion)
    # Ordeno de mejor a peor según R2
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
    Esta función es especial. Cuando hay un grupo, no quiero predecir a la persona exacta,
    sino ver si "la opinión general del grupo" y "la distribución predicha" se parecen.
    Uso Kullback-Leibler y Jensen-Shannon (estadística avanzada de distribuciones).
    """
    if len(y_true) < 2: return 0, 0, 0 
    
    dme = abs(np.mean(y_true) - np.mean(y_pred)) # Diferencia de medias simple
    
    # Creo 'cajas' (bins) desde el valor mínimo de la escala ISO al máximo (-1 a 1)
    bins = np.linspace(-1, 1, 11)
    
    # Calculo las frecuencias (histogramas) de los datos reales y las predicciones
    hist_true, _ = np.histogram(y_true, bins=bins, density=True)
    hist_pred, _ = np.histogram(y_pred, bins=bins, density=True)
    
    # Añado un valor mínimo (1e-10) para no dividir entre 0 y romper el cálculo de entropía
    p = (hist_true + 1e-10) / np.sum(hist_true + 1e-10)
    q = (hist_pred + 1e-10) / np.sum(hist_pred + 1e-10)
    
    kl = entropy(p, q)            # Divergencia KL: cuánta info pierdo al aproximar
    js = jensenshannon(p, q)      # Divergencia JS: versión simétrica y acotada de KL (0 es idéntico)
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
            "JS Divergence (0=Perfecto)": js, # La métrica reina aquí. Más cerca de 0, mejor.
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
    Junto la tabla de resultados de individuos con la de grupos por el nombre del modelo
    y guardo el resultado en un CSV para el informe de mi TFG.
    """
    print("\n" + "="*65)
    print(f" 🌍 TABLA MAESTRA DE RENDIMIENTO PARA: {target_name.upper()} ")
    print("="*65)

    if df_singles.empty or df_grupos.empty: return pd.DataFrame()

    # Hago un merge (como un BUSCARV en Excel) uniendo por la columna 'Modelo'
    df_global = pd.merge(df_singles, df_grupos, on='Modelo', how='inner')
    # Ordeno la tabla general fijándome en quién tiene menor MAE
    df_global = df_global.sort_values(by='MAE (Error Absoluto)', ascending=True).reset_index(drop=True)

    print(df_global.to_string(index=False))
    
    # Guardo mi tablita bien formateada en la carpeta del proyecto
    carpeta_resultados = ruta_script / 'Graficas_IA_TFG' / 'Resultados_Metricas'
    carpeta_resultados.mkdir(parents=True, exist_ok=True) # Crea la carpeta si no existe
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
    1. Entrenamiento: Predice sobre lo que ya ha visto (es fácil sacar buena nota).
    2. Validación Cruzada: Simulación interna de datos nuevos (nota más realista).
    3. Test (Groningen): Generalización pura a la nueva ciudad (la prueba de fuego).
    
    IMPORTANTE: Filtramos solo a los individuos solitarios (Singles) para que el R2/MAE/RMSE 
    sea matemáticamente justo y no se contamine con el comportamiento grupal.
    """
    print(f"\n" + "⚔️"*25)
    print(f" AUDITORÍA DE FASES DE APRENDIZAJE: {target_name.upper()} ")
    print("⚔️"*25)
    
    carpeta_auditorias = ruta_script / 'Graficas_IA_TFG' / 'Auditorias_Individuales'
    carpeta_auditorias.mkdir(parents=True, exist_ok=True)
    
    # Extraigo solo a las personas individuales (mask == 0) de la matriz de entrenamiento
    mask_singles_train = (mask_train == 0)
    X_train_singles = X_train[mask_singles_train]
    y_train_singles = y_train[mask_singles_train]
    
    # Hago lo mismo para la matriz del examen final (Test)
    mask_singles_test = (mask_test == 0)
    X_test_singles = X_test[mask_singles_test]
    y_test_singles = y_test[mask_singles_test]

    for nombre_modelo, pipeline_optimo in mejores_modelos.items():
        print(f"   📊 Dibujando auditoría de fases para: {nombre_modelo}...")
        
        # 1. PREDICCIONES EN LAS TRES FASES
        
        # Fase 1: Entrenamiento puro (El modelo "recita de memoria")
        pred_train = pipeline_optimo.predict(X_train_singles)
        
        # Fase 2: Validación cruzada 
        # Usamos cv=5 para predecir a los propios individuos del Train, pero de forma oculta (es más honesto)
        pred_val = cross_val_predict(pipeline_optimo, X_train_singles, y_train_singles, cv=5)
        
        # Fase 3: Test (Groningen, el choque con la realidad)
        pred_test = pipeline_optimo.predict(X_test_singles)
        
        # Estructuro los datos en una lista para convertirlos en DataFrame fácilmente luego
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
        
        # 2. DIBUJAR LA GRÁFICA 2x2
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Auditoría de Aprendizaje y Generalización\nModelo: {nombre_modelo} ({target_name})', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        metricas_unicas = ['R2 (Nota /100%)', 'MAE (Error Absoluto)', 'RMSE (Error Medio)', 'MSE (Error² Medio)']
        colores_fases = ["#3498db", "#f39c12", "#e74c3c"] # Azul (Train), Naranja (Val), Rojo (Test)
        
        for i, ax in enumerate(axes.flatten()):
            metrica = metricas_unicas[i]
            # Filtro los datos para dibujar solo la métrica actual del sub-gráfico
            df_filtrado = df_plot[df_plot['Métrica'] == metrica]
            
            sns.barplot(data=df_filtrado, x='Fase', y='Valor', ax=ax, palette=colores_fases, edgecolor='black', linewidth=1.5)
            
            # Añado texturas (Hatches) para que la gráfica se entienda si se imprime en blanco y negro en el documento
            hatches = ['///', '...', 'xx']
            for j, bar in enumerate(ax.patches):
                bar.set_hatch(hatches[j % len(hatches)])
                
            # Escribo el valor numérico exacto encima de cada barra
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', padding=5, fontsize=11, fontweight='bold')
                
            ax.set_title(metrica, fontsize=14, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            
            # Limites dinámicos del eje Y (Para dejar espacio al número que escribí encima de la barra)
            y_max = df_filtrado['Valor'].max()
            if df_filtrado['Valor'].min() < 0: # El R2 puede ser negativo
                ax.set_ylim(df_filtrado['Valor'].min() * 1.15, y_max * 1.25)
            else:
                ax.set_ylim(0, y_max * 1.25)
                
        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Ajuste para que el título general no se solape
        
        # 3. GUARDAR
        # Limpio el nombre por si el modelo tuviese caracteres raros
        nombre_limpio = nombre_modelo.replace('/', '_').replace(' ', '_').replace('.', '')
        ruta_img = carpeta_auditorias / f"Auditoria_Fases_{nombre_limpio}_{target_name}.png"
        plt.savefig(ruta_img, dpi=300, bbox_inches='tight')
        plt.close() # Cierro la figura para liberar memoria RAM

# Llamo a la función de auditoría para analizar el comportamiento con ambas variables objetivo
generar_auditoria_individual_completa(mejores_modelos_p, X_train, y_train_p, X_test, y_test_p, mask_train_grupos, mask_test_grupos, "ISOPleasant")
generar_auditoria_individual_completa(mejores_modelos_e, X_train, y_train_e, X_test, y_test_e, mask_train_grupos, mask_test_grupos, "ISOEventful")
print(f"✅ Auditorías de Fases finalizadas con éxito.")
# ==============================================================================
# 8.9. COMPARATIVA GLOBAL (TODAS LAS IAs AGRUPADAS) - VERSIÓN MSE
# ==============================================================================
def generar_comparativa_global_ias(df_resumen_global, target_name):
    """
    Genera una gráfica de barras comparando MAE, RMSE y MSE para todos los modelos.
    """
    print(f"\n" + "🌟"*20)
    print(f" DIBUJANDO COMPARATIVA GLOBAL PARA: {target_name.upper()} ")
    print("🌟"*20)

    if df_resumen_global.empty:
        print(f"   ⚠️ No hay datos globales para {target_name}. Saltando gráfica.")
        return

    # 1. SELECCIÓN DE MÉTRICAS A COMPARAR
    metricas_cols = ['MAE (Error Absoluto)', 'RMSE (Error Raíz Cuadrada)', 'MSE (Error² Medio)']
    
    # Comprobación de seguridad para evitar errores si la columna tiene otro nombre
    columnas_presentes = [col for col in metricas_cols if col in df_resumen_global.columns]
    
    if len(columnas_presentes) == 0:
        print("   ❌ ERROR: No se encontraron las métricas de error en el DataFrame.")
        return

    # 2. TRANSFORMACIÓN DE DATOS (Melt)
    df_melted = df_resumen_global.melt(
        id_vars='Modelo', 
        value_vars=columnas_presentes, 
        var_name='Métrica', 
        value_name='Valor'
    )
    
    # 3. DISEÑO DE LA GRÁFICA
    plt.figure(figsize=(15, 8))
    
    # Colores: Azul para MAE, Naranja para RMSE, Verde para MSE
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
    
    # Estética y Títulos
    plt.title(f'Rendimiento Global de los Algoritmos de IA en Groningen\nObjetivo: {target_name.upper()}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=15, ha='right', fontsize=12, fontweight='bold')
    plt.xlabel('Algoritmos de Inteligencia Artificial', fontsize=13, fontweight='bold', labelpad=15)
    plt.ylabel('Puntuación de la Métrica (Más bajo es MEJOR)', fontsize=13, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.legend(title='Métricas de Evaluación', loc='upper right', framealpha=0.9, fontsize=11)
    
    # 4. AÑADIR NÚMEROS A LAS BARRAS
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=5, fontsize=10, rotation=90)
        
    # 5. AJUSTE DE EJES (Damos espacio para las etiquetas rotadas)
    y_max = df_melted['Valor'].max()
    plt.ylim(0, y_max * 1.4)

    plt.tight_layout()
    
    # 6. GUARDAR LA IMAGEN
    carpeta_comparativas = ruta_script / 'Graficas_IA_TFG' / 'Comparativas_Globales'
    carpeta_comparativas.mkdir(parents=True, exist_ok=True)
    
    ruta_guardado = carpeta_comparativas / f"Comparativa_Global_IAs_{target_name}.png"
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Gráfica comparativa global guardada en: {ruta_guardado.name}")

# LLAMAMOS A LA FUNCIÓN PARA AMBAS COORDENADAS
# Le paso las tablas resumen 'df_global_p' y 'df_global_e' que guardé en la sección 8.7
generar_comparativa_global_ias(df_global_p, "ISOPleasant")
generar_comparativa_global_ias(df_global_e, "ISOEventful")

# ==============================================================================
# 9. EXTRACCIÓN DE INTERPRETABILIDAD (PESOS Y GINI IMPORTANCE)
# ==============================================================================

def extraer_e_imprimir_importancias(mejores_modelos, X_train, y_train, target_name):
    """
    Entra en "el cerebro" de cada IA y extrae qué características del sonido pesaron más en sus decisiones.
    No todos los modelos "hablan el mismo idioma", por eso separo por tipos:
    - Modelos Lineales: Nos dan Coeficientes (+/- influye).
    - Árboles (RF, XGB): Nos dan Gini Importance (% de uso del dato en los árboles).
    - Cajas Negras (SVR, GPR): Usamos Permutation Importance (barajamos los datos y vemos cuánto sube el error).
    """
    print(f"\n" + "="*50)
    print(f" EXTRAYENDO TOP VARIABLES INDIVIDUALES: {target_name.upper()} ")
    print("="*50)
    
    carpeta_pesos = ruta_script / 'Graficas_IA_TFG' / 'Interpretabilidad'
    carpeta_pesos.mkdir(parents=True, exist_ok=True)
    
    # Guardo la lista de nombres de las variables (X) para ponerle etiqueta a los pesos
    nombres_variables = X_train.columns.tolist()
    
    for nombre_modelo, pipeline_optimo in mejores_modelos.items():
        print(f"   📊 Analizando los secretos de: {nombre_modelo}...")
        importancias = None
        es_coeficiente = False
        
        # Saco el modelo matemático desnudo del pipeline (ignorando el StandardScaler)
        modelo_real = pipeline_optimo.named_steps['model']
        # Necesito escalar los datos X a mano para pasárselos a las "cajas negras" después
        X_train_escalado = pipeline_optimo.named_steps['scaler'].transform(X_train)
        
        # 1. SI ES UN ÁRBOL (RandomForest, XGBoost)
        if nombre_modelo in ["RandomForest", "XGBoost"]:
            importancias = modelo_real.feature_importances_
            
        # 2. SI ES UN MODELO LINEAL (LinearRegression, Lasso)
        elif nombre_modelo in ["LinearRegression", "Lasso"]:
            importancias = modelo_real.coef_
            es_coeficiente = True
            
        # 3. SI ES UNA CAJA NEGRA (SVR, GaussianProcess)
        else:
            print("      🧠 Aplicando Permutation Importance (barajando variables)...")
            # Truco para descifrar cajas negras: barajo aleatoriamente una columna 5 veces. 
            # Si el modelo comete muchos errores de repente, ¡esa columna era importantísima!
            resultado_perm = permutation_importance(
                modelo_real, X_train_escalado, y_train, 
                n_repeats=5, random_state=42, scoring='neg_mean_absolute_error'
            )
            # Me quedo con la media del impacto que ha tenido barajar cada variable
            importancias = resultado_perm.importances_mean
            
        # --- PROCESADO DE DATOS PARA LA GRÁFICA ---
        # Creo un dataframe temporal emparejando el nombre de la variable acústica con su valor de importancia
        df_imp = pd.DataFrame({'Variable': nombres_variables, 'Valor': importancias})
        
        # Filtramos para quedarnos solo con el top de características relevantes (10-15)
        if es_coeficiente:
            # En modelos lineales, una variable con coeficiente -3 es tan importante como una con +3.
            # Por eso saco el valor absoluto para ordenar por "fuerza" y no por dirección.
            df_imp['Fuerza'] = df_imp['Valor'].abs()
            df_imp = df_imp[df_imp['Fuerza'] > 0.0].sort_values(by='Fuerza', ascending=False).head(15)
        else:
            # En árboles y permutaciones, no hay negativos. Un % más alto = más importancia
            # Limpio las variables que directamente la IA ha ignorado (< 0.001)
            df_imp = df_imp[df_imp['Valor'] > 0.001].sort_values(by='Valor', ascending=False).head(15)
            
        if len(df_imp) == 0: 
            print(f"      ⚠️ {nombre_modelo} ha anulado todas las variables. Saltando gráfica.")
            continue
            
        # --- DIBUJO DE LA GRÁFICA (Tumbada horizontal para leer bien los nombres) ---
        plt.figure(figsize=(10, 6))
        
        if es_coeficiente:
            # Pongo colores lógicos: Rojo si el sonido resta agrado, Verde si el sonido suma agrado
            colores = ["#e74c3c" if val < 0 else "#2ecc71" for val in df_imp['Valor']]
            ax = sns.barplot(data=df_imp, x='Valor', y='Variable', palette=colores, edgecolor='black')
            plt.xlabel("Peso del Coeficiente (Impacto direccional)")
            plt.title(f"Top 15 Factores Clave (Coeficientes) - {nombre_modelo}\n({target_name})")
        else:
            # Aquí uso un gradiente magma (de oscuro a claro) porque solo importa el volumen de impacto
            ax = sns.barplot(data=df_imp, x='Valor', y='Variable', palette="magma", edgecolor='black')
            plt.xlabel("Nivel de Importancia (Impacto en las decisiones de la IA)")
            plt.title(f"Top 15 Factores Clave (Importancia) - {nombre_modelo}\n({target_name})")
            
        plt.ylabel("Parámetros Acústicos")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Lógica matemática pesada para colocar el texto flotante de los números sin que tape la barra
        for i, val in enumerate(df_imp['Valor']):
            texto = f"{val:.3f}" if es_coeficiente else f"{val:.4f}" 
            # ¡AQUÍ ESTÁ LA CORRECCIÓN MÁGICA! uso abs() estándar de Python porque val es un número suelto, no una Serie
            offset = (df_imp['Valor'].max() * 0.02) if val >= 0 else -(abs(df_imp['Valor'].min()) * 0.02)
            ha_align = 'left' if val >= 0 else 'right'
            ax.text(val + offset, i, texto, va='center', ha=ha_align, fontsize=10, fontweight='bold')
            
        # Ajustar límites del eje X horizontal para que el texto numérico nunca se salga del margen de la pantalla
        x_min, x_max = plt.xlim()
        rango = x_max - x_min
        plt.xlim(x_min - (rango*0.1) if x_min < 0 else 0, x_max + (rango*0.15))
        
        plt.tight_layout()
        
        # Guardo la gráfica de pesos de este modelo concreto
        ruta_img = carpeta_pesos / f"Pesos_{nombre_modelo.replace('.', '')}_{target_name}.png"
        plt.savefig(ruta_img, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✅ Gráfica generada: {ruta_img.name}")

# LLAMAMOS A LA FUNCIÓN (Le paso y_train porque la Permutation Importance necesita simular predicciones)
extraer_e_imprimir_importancias(mejores_modelos_p, X_train, y_train_p, "ISOPleasant")
extraer_e_imprimir_importancias(mejores_modelos_e, X_train, y_train_e, "ISOEventful")

# ==============================================================================
# 10. ENSAMBLAJE (ENVERGUDURA GLOBAL): LA "SÚPER PREDICCIÓN"
# ==============================================================================
def generar_super_prediccion_ensamblada(mejores_modelos, df_rank_singles, X_test, y_test_real, target_name):
    """
    El colofón del proyecto. La teoría dice que un modelo lineal (muy estable pero tonto) 
    y un modelo complejo (muy listo pero que sobre-reacciona) se corrigen mutuamente si haces la media.
    Aquí construyo un "Frankenstein" con el mejor de cada tipo.
    """
    print("\n" + "🌟"*25)
    print(f" GENERANDO SÚPER PREDICCIÓN ENSAMBLADA: {target_name.upper()} ")
    print("🌟"*25)
    
    if df_rank_singles.empty: return
        
    lineales = ["LinearRegression", "Lasso"]
    complejos = ["RandomForest", "XGBoost", "SVR", "GaussianProcess"]
    
    # Divido la tabla de rankings de la fase 8.5 en dos ligas
    df_lineales = df_rank_singles[df_rank_singles['Modelo'].isin(lineales)]
    df_complejos = df_rank_singles[df_rank_singles['Modelo'].isin(complejos)]
    
    if df_lineales.empty or df_complejos.empty: return
        
    # Elijo al campeón de cada liga basándome en quién tiene el menor error absoluto (MAE)
    mejor_lineal = df_lineales.sort_values(by='MAE (Error Absoluto)').iloc[0]['Modelo']
    mejor_complejo = df_complejos.sort_values(by='MAE (Error Absoluto)').iloc[0]['Modelo']
    
    print(f"   🏆 Campeón Lineal: {mejor_lineal}")
    print(f"   🏆 Campeón Complejo: {mejor_complejo}")
    
    # Que ambos campeones hagan su predicción sobre el examen de Groningen
    prediccion_lineal = mejores_modelos[mejor_lineal].predict(X_test)
    prediccion_compleja = mejores_modelos[mejor_complejo].predict(X_test)
    
    # Promedio simple (soft-voting / ensamblado de promedios)
    super_prediccion = (prediccion_lineal + prediccion_compleja) / 2.0
    
    # Calculo las métricas de mi Súper Modelo a ver si ha superado a sus "padres" por separado
    mae_super = mean_absolute_error(y_test_real, super_prediccion)
    rmse_super = np.sqrt(mean_squared_error(y_test_real, super_prediccion))
    r2_super = r2_score(y_test_real, super_prediccion)
    
    print(f"   📊 SÚPER MODELO -> R2: {r2_super:.4f} | MAE: {mae_super:.4f} | RMSE: {rmse_super:.4f}")
    
    # Dibujo la dispersión final (Scatter Plot) de Realidad vs Predicción
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test_real, y=super_prediccion, alpha=0.6, color='#8e44ad', s=60, edgecolor='white')
    
    # Trazo la línea diagonal perfecta (Si un punto cae aquí, el modelo ha acertado el 100%)
    limite_min = min(y_test_real.min(), super_prediccion.min()) - 0.1
    limite_max = max(y_test_real.max(), super_prediccion.max()) + 0.1
    plt.plot([limite_min, limite_max], [limite_min, limite_max], color='red', linestyle='--', linewidth=2)
    
    plt.title(f'Realidad vs Súper Predicción Ensamblada\n({mejor_lineal} + {mejor_complejo}) - {target_name}', fontsize=14, pad=15)
    plt.xlabel('Respuesta Real (Encuestas)')
    plt.ylabel('Súper Predicción (IA Ensamblada)')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Escribo la nota final de mi modelo híbrido en un cuadrito flotante en la esquina superior izquierda
    texto_box = f"$R^2$ = {r2_super:.4f}\nMAE = {mae_super:.4f}"
    plt.text(0.05, 0.95, texto_box, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
             
    plt.tight_layout()
    carpeta_ensamble = ruta_script / 'Graficas_IA_TFG' / 'Ensamblaje'
    carpeta_ensamble.mkdir(parents=True, exist_ok=True)
    plt.savefig(carpeta_ensamble / f"Super_Prediccion_{target_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

generar_super_prediccion_ensamblada(mejores_modelos_p, df_singles_p, X_test, y_test_p, "ISOPleasant")
generar_super_prediccion_ensamblada(mejores_modelos_e, df_singles_e, X_test, y_test_e, "ISOEventful")

print("\n" + "🚀"*20)
print(" PIPELINE DE MACHINE LEARNING FINALIZADO COMPLETAMENTE ")
print("🚀"*20)