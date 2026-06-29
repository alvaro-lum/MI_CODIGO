# ==========================================
# Codigo 4: aprendizaje máquina general sin distinción de grupos.
# Train --> Todas las ciudades. Test --> todas las ciudades.
# ==========================================
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# ==========================================
# 1. LIBRERÍAS DE EVALUACIÓN Y MÉTRICAS
# ==========================================
# Las típicas para saber cuánto se aleja mi predicción de la realidad (MSE, MAE) y si explico bien los datos (R2)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Estas dos me sirven para evaluar a la población en conjunto. Comparan si "la montaña" (distribución) 
# de respuestas reales se parece a la "montaña" de mis predicciones.
from scipy.stats import entropy # Para calcular KL Divergence (Kullback-Leibler)
from scipy.spatial.distance import jensenshannon # Para calcular JS Divergence (Jensen-Shannon)

# ==========================================
# 2. LIBRERÍAS DE MODELOS DE MACHINE LEARNING
# ==========================================
# Mi "plantilla" de ML, ordenadas de más simples a más complejas:
from sklearn.linear_model import LinearRegression, Lasso # Modelos lineales (fáciles de interpretar)
from sklearn.ensemble import RandomForestRegressor       # Un bosque lleno de árboles de decisión
from sklearn.svm import SVR                              # Support Vector Regression (ajusta un tubo de tolerancia)
from sklearn.gaussian_process import GaussianProcessRegressor # Modelo probabilístico (Bayesiano)
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C # Funciones matemáticas para el modelo Gaussiano
import xgboost as xgb                                    # El modelo estrella basado en árboles potenciados

# ==========================================
# 3. VALIDACIÓN CRUZADA Y OPTIMIZACIÓN (GRID SEARCH)
# ==========================================
# Herramientas para que el modelo entrene probando configuraciones y no haga trampas memorizando
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict

# ==========================================
# 4. LIBRERÍAS DE VISUALIZACIÓN Y PIPELINE
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns # Para que las gráficas queden mucho más bonitas
# Pipeline me permite encadenar pasos (1º limpiar/escalar, 2º predecir) sin que se me mezclen los datos
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler # Normaliza los datos (pone todas las variables a la misma escala)

# Ocultamos avisos técnicos (los típicos textos rojos larguísimos) para mantener la terminal limpia
warnings.filterwarnings('ignore')

# ==========================================
# 5. CONFIGURACIÓN DE RUTAS Y CARGA DE DATOS
# ==========================================
# Guardo la ruta donde está este script para cargar los CSVs de forma relativa (y que funcione en cualquier PC)
ruta_script = Path(__file__).parent
ciudades = ['Londres', 'Venecia', 'Granada', 'Groningen']
dfs_ciudades = {}

print("\n" + "="*50)
print(" INICIANDO CARGA Y PREPROCESAMIENTO DE DATOS ")
print("="*50)

# Bucle para leer el CSV de cada ciudad y guardarlo en mi diccionario 'dfs_ciudades'
for ciudad in ciudades:
    nombre_carpeta = f"Graficas_{ciudad}_TFG"
    nombre_archivo = f"ML_Datos_{ciudad}.csv"
    ruta_csv = ruta_script / nombre_carpeta / 'General' / nombre_archivo
    
    if ruta_csv.exists():
        # Uso decimal=',' y sep=';' porque exporté los datos con formato español/europeo
        df_temp = pd.read_csv(ruta_csv, sep=';', decimal=',', low_memory=False)
        # Paso todos los nombres de columnas a minúsculas para no liarme con mayúsculas luego
        df_temp.columns = df_temp.columns.str.lower()
        # Añado una etiqueta para saber de qué ciudad viene cada fila antes de mezclarlo todo
        df_temp['origen_ciudad'] = ciudad
        
        dfs_ciudades[ciudad] = df_temp
        print(f"✅ {ciudad} cargada: {len(df_temp)} encuestas.")
    else:
        print(f"❌ ERROR: No se encuentra el archivo para {ciudad}.")

print("\nCarga finalizada. Datos listos en memoria.")

# ==============================================================================
# 6. LIMPIEZA Y SEPARACIÓN GLOBAL (95% TRAIN / 5% TEST)
# ==============================================================================
print("\n" + "="*50)
print(" PREPARACIÓN DE MATRICES (TRAIN vs TEST GLOBAL) ")
print("="*50)

def aislar_variables_ml(df):
    """
    Aquí limpio el DataFrame crudo. Separó lo que quiero predecir (Y) de las variables 
    acústicas/contextuales (X) que uso para adivinarlo. También borro columnas que hacen trampa.
    """
    if 'isopleasant' not in df.columns or 'isoeventful' not in df.columns:
        raise ValueError("CRÍTICO: Faltan las columnas 'isopleasant' o 'isoeventful' en el CSV.")
        
    # Extraigo mis dos variables objetivo (Las coordenadas del circumplex model)
    Y_p = df['isopleasant'].fillna(0)
    Y_e = df['isoeventful'].fillna(0)
    
    # Lista negra de columnas: IDs, cosas geográficas que el ML no debe saber, 
    # y las respuestas directas de la encuesta que desvelarían el resultado (eso sería hacer trampa)
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
    
    # Solo borro las prohibidas que de verdad existan en este DataFrame para evitar errores
    columnas_a_borrar = [col for col in columnas_prohibidas if col in df.columns]
    X = df.drop(columns=columnas_a_borrar).fillna(0)
    # Me aseguro al 100% de que la matriz X sea numérica (los ML no entienden de texto)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return X, Y_p, Y_e

# === NUEVA ESTRATEGIA DE EXAMEN (GLOBAL 95/5) ===
# En lugar de aislar una ciudad entera, juntamos TODAS las encuestas de las 4 ciudades.
# El modelo entrenará con el 95% de ellas elegidas al azar, y se examinará con el 5% restante.

print(f"🎯 Variables objetivo configuradas: ISOPLEASANT e ISOEVENTFUL")

# 1. Juntar todos los DataFrames de las ciudades en uno solo (df_global_crudo)
lista_todos_datos = [df for ciudad, df in dfs_ciudades.items()]
df_global_crudo = pd.concat(lista_todos_datos, ignore_index=True)

# 2. Aplicar la limpieza general para sacar las X y las Y de todo el planeta
X_global, Y_global_p, Y_global_e = aislar_variables_ml(df_global_crudo)

# 3. Hacer la partición 95% / 5%
# Usamos random_state=42 para que siempre elija a los mismos alumnos para el examen
# Importamos la herramienta necesaria (asegúrate de tenerla arriba en el código, 
# aunque aquí la pongo por si acaso: from sklearn.model_selection import train_test_split)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train_p, y_test_p, y_train_e, y_test_e = train_test_split(
    X_global, Y_global_p, Y_global_e, 
    test_size=0.05, 
    random_state=42
)

print(f"📚 SET DE ENTRENAMIENTO (Train 95%): {len(X_train)} encuestas (Todas las ciudades mezcladas).")
print(f"   -> Variables predictoras (Columnas X resultantes): {len(X_train.columns)}")
print(f"📝 SET DE EXAMEN (Test 5%): {len(X_test)} encuestas elegidas al azar.")

# ==============================================================================
# 7. CONSTRUCCIÓN DEL PIPELINE Y GRID SEARCH (OPTIMIZACIÓN DE MODELOS)
# ==============================================================================
print("\n" + "="*50)
print(" INICIANDO PIPELINE Y BÚSQUEDA DE HIPERPARÁMETROS ")
print("="*50)

def optimizar_modelo_gridsearch(X, y, tipo_modelo):
    """
    En lugar de usar el modelo por defecto, esta función prueba muchas configuraciones 
    diferentes (hiperparámetros) y se queda con la que comete menos errores (MAE).
    """
    print(f"   ⚙️ Configurando Grid Search para: {tipo_modelo}")
    # Divido el entrenamiento en 5 trozos para auto-evaluarse sin memorizar
    cv_kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Según el modelo que toque, defino qué parámetros quiero que el programa pruebe
    if tipo_modelo == "LinearRegression":
        # Primero siempre escalo (StandardScaler) y luego aplico el modelo
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
        grid_params = {} # La regresión lineal es sosa, no tiene parámetros que tunear
        
    elif tipo_modelo == "Lasso":
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', Lasso(random_state=42))])
        grid_params = {'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]} # Pruebo cuánto penaliza las variables inútiles
        
    elif tipo_modelo == "RandomForest":
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(random_state=42))])
        grid_params = {
            'model__n_estimators': [50, 100, 200], # Cuántos árboles crecen
            'model__max_depth': [None, 10, 20],    # Cuánto de profundos pueden ser
            'model__min_samples_split': [2, 5, 10]
        }
        
    elif tipo_modelo == "XGBoost":
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', xgb.XGBRegressor(random_state=42, objective='reg:squarederror'))])
        grid_params = {
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7],
            'model__subsample': [0.8, 1.0]
        }
        
    elif tipo_modelo == "SVR":
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', SVR())])
        grid_params = {
            'model__kernel': ['rbf', 'linear'],
            'model__C': [0.1, 1.0, 10.0],
            'model__epsilon': [0.01, 0.1, 0.5]
        }
        
    elif tipo_modelo == "GaussianProcess":
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', GaussianProcessRegressor(random_state=42))])
        grid_params = {
            'model__kernel': [1.0 * RBF(1.0), C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))],
            'model__alpha': [1e-2, 1e-1, 1.0]
        }
    else:
        raise ValueError(f"Modelo {tipo_modelo} no soportado.")

    # Configuro el buscador: quiero que encuentre la mejor combinación minimizando el error absoluto (negativo porque sklearn intenta maximizar siempre)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid_params,
        cv=cv_kfold,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,  # Usa todos los núcleos del PC para acabar antes
        verbose=1   
    )
    
    print(f"   🚀 Iniciando entrenamiento...")
    grid_search.fit(X, y)
    print(f"   ✅ Mejor configuración: {grid_search.best_params_}")
    print('-----------------------------------------------------')
    
    # Retorno la "versión prime" del modelo, ya entrenada y optimizada
    return grid_search.best_estimator_

# ==============================================================================
# 8. EJECUCIÓN DEL ENTRENAMIENTO PARA TODOS LOS MODELOS
# ==============================================================================
modelos_a_probar = [
    "LinearRegression", "Lasso", "RandomForest", 
    "XGBoost", "SVR", "GaussianProcess"
]

# Aquí voy a guardar los modelos ya entrenados para no perderlos
mejores_modelos_p = {} 
mejores_modelos_e = {} 

print("\n--- ENTRENANDO EXPERTOS EN 'ISOPleasant' ---")
for nombre in modelos_a_probar:
    mejores_modelos_p[nombre] = optimizar_modelo_gridsearch(X_train, y_train_p, nombre)

print("\n--- ENTRENANDO EXPERTOS EN 'ISOEventful' ---")
for nombre in modelos_a_probar:
    mejores_modelos_e[nombre] = optimizar_modelo_gridsearch(X_train, y_train_e, nombre)

print("\n🎉 Todos los modelos han sido entrenados y optimizados con éxito.")

# ==============================================================================
# 8.5 EVALUACIÓN GLOBAL (RANKING UNIVERSAL)
# ==============================================================================
def calcular_metricas_distribucion(y_true, y_pred):
    """
    Calcula métricas de consenso poblacional.
    Me dice si mi ML ha capturado bien "la vibra general" de la ciudad,
    comparando los histogramas de respuestas reales vs predichas.
    """
    if len(y_true) < 2: return 0, 0, 0 
    # Diferencia entre la media real y la predicha
    dme = abs(np.mean(y_true) - np.mean(y_pred))
    
    # Corto el eje ISO (-1 a 1) en 10 cajones (bins)
    bins = np.linspace(-1, 1, 11)
    
    # Construyo los histogramas (density=True para que sea porcentaje y no conteo absoluto)
    hist_true, _ = np.histogram(y_true, bins=bins, density=True)
    hist_pred, _ = np.histogram(y_pred, bins=bins, density=True)
    
    # Sumo 1e-10 para no romper la fórmula si hay algún cajón vacío (no se puede dividir por cero)
    p = (hist_true + 1e-10) / np.sum(hist_true + 1e-10)
    q = (hist_pred + 1e-10) / np.sum(hist_pred + 1e-10)
    
    kl = entropy(p, q)           # Cuánta información se pierde
    js = jensenshannon(p, q)     # Qué tan diferentes son las montañas (0 es que son clones)
    return kl, js, dme

def evaluar_ciudad_completa(mejores_modelos, X_test, y_test_real, target_name):
    """
    Evalúa a todos los ciudadanos de Groningen de golpe y genera un DataFrame 
    ordenado de mejor a peor modelo.
    """
    print("\n" + "-"*65)
    print(f" 🌍 RANKING DE RENDIMIENTO (Población Universal): {target_name.upper()} ")
    print("-"*65)
    
    resultados_evaluacion = []
    
    for nombre_modelo, pipeline_optimo in mejores_modelos.items():
        # Examen final para este modelo
        predicciones = pipeline_optimo.predict(X_test)
        
        # Saco todas mis métricas
        mae = mean_absolute_error(y_test_real, predicciones)
        rmse = np.sqrt(mean_squared_error(y_test_real, predicciones))
        mse = mean_squared_error(y_test_real, predicciones)
        r2 = r2_score(y_test_real, predicciones)
        kl, js, dme = calcular_metricas_distribucion(y_test_real, predicciones)
        
        resultados_evaluacion.append({
            "Modelo": nombre_modelo,
            "MAE": mae,
            "RMSE": rmse,
            "MSE": mse,
            "R2 (Varianza)": r2,
            "JS Divergence": js,
            "KL Divergence": kl,
            "DME": dme
        })
        
    df_resultados = pd.DataFrame(resultados_evaluacion)
    # Ordeno el ranking fijándome en quién tiene menor error MAE
    df_resultados = df_resultados.sort_values(by="MAE (Error Absoluto)", ascending=True).reset_index(drop=True)
    df_resultados = df_resultados.round(4)
    
    print(df_resultados.to_string(index=False))
    
    # Guardo este ranking en un CSV para luego meterlo en la memoria del TFG
    carpeta_resultados = ruta_script / 'Graficas_IA_AM2_TFG' / 'Resultados_Metricas'
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    df_resultados.to_csv(carpeta_resultados / f"Resumen_Universal_{target_name}.csv", sep=';', index=False, decimal=',')
    
    return df_resultados

df_global_p = evaluar_ciudad_completa(mejores_modelos_p, X_test, y_test_p, "ISOPleasant")
df_global_e = evaluar_ciudad_completa(mejores_modelos_e, X_test, y_test_e, "ISOEventful")

# ==============================================================================
# 8.6 AUDITORÍA INDIVIDUAL (TRAIN vs VAL vs TEST) -> EL NUEVO BLOQUE
# ==============================================================================
def generar_auditoria_individual(mejores_modelos, X_train, y_train, X_test, y_test, target_name):
    """
    Genera una gráfica 2x2 por modelo comprobando si el ML ha aprendido o solo memorizado.
    1. Train (Memorización): Nota sobre lo que ya estudió.
    2. Validation (Robustez Interna): Nota de simulacros internos.
    3. Test (Generalización a Groningen): La nota del examen final real.
    """
    print(f"\n" + "⚔️"*20)
    print(f" AUDITORÍA INDIVIDUAL (Train vs Val vs Test) - {target_name.upper()} ")
    print("⚔️"*20)
    
    carpeta_auditoria = ruta_script / 'Graficas_IA_AM2_TFG' / 'Auditorias_Individuales'
    carpeta_auditoria.mkdir(parents=True, exist_ok=True)
    
    for nombre, pipeline_optimo in mejores_modelos.items():
        print(f"   📊 Generando gráficas de auditoría para: {nombre}...")
        
        # 1. Lanzo las predicciones simulando las 3 situaciones de estrés del modelo
        pred_train = pipeline_optimo.predict(X_train)
        pred_val = cross_val_predict(pipeline_optimo, X_train, y_train, cv=5)
        pred_test = pipeline_optimo.predict(X_test)
        
        fases = [
            ('1. Train\n(3 Ciudades)', y_train, pred_train),
            ('2. Validation\n(CV-5 Pliegues)', y_train, pred_val),
            ('3. Test\n(Examen)', y_test, pred_test)
        ]
        
        # Recopilo los datos para graficarlos
        filas = []
        for fase_nombre, y_real, y_pred in fases:
            mse = mean_squared_error(y_real, y_pred)
            filas.extend([
                {'Métrica': 'R2', 'Valor': r2_score(y_real, y_pred), 'Fase': fase_nombre},
                {'Métrica': 'MAE', 'Valor': mean_absolute_error(y_real, y_pred), 'Fase': fase_nombre},
                {'Métrica': 'RMSE', 'Valor': np.sqrt(mse), 'Fase': fase_nombre},
                {'Métrica': 'MSE', 'Valor': mse, 'Fase': fase_nombre}
            ])
            
        df_plot = pd.DataFrame(filas)
        
        # 2. Diseñar la gráfica 2x2 (un subplot para cada métrica)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Auditoría de Generalización: {nombre}\n({target_name.upper()})', fontsize=18, fontweight='bold', y=0.98)
        
        metricas_unicas = ['R2', 'MAE', 'RMSE', 'MSE']
        # Uso semáforo de colores: Azul (bien/cómodo), Naranja (aviso), Rojo (fuego real/test)
        colores = ["#3498db", "#f39c12", "#e74c3c"] 
        
        for i, ax in enumerate(axes.flatten()):
            metrica = metricas_unicas[i]
            df_filtrado = df_plot[df_plot['Métrica'] == metrica]
            
            sns.barplot(data=df_filtrado, x='Fase', y='Valor', ax=ax, palette=colores, edgecolor='black', linewidth=1.2)
            
            # Texturas (Hatches) para hacerlo más académico y legible en blanco y negro
            hatches = ['///', '...', 'xx']
            for j, bar in enumerate(ax.patches):
                bar.set_hatch(hatches[j % len(hatches)])
            
            # Pongo el numerito exacto encima de cada barra
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', padding=3, fontsize=11, fontweight='bold')
            
            ax.set_title(metrica, fontsize=14, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            # Ajuste dinámico de los ejes Y para que el texto de arriba no se corte
            y_max = df_filtrado['Valor'].max()
            if df_filtrado['Valor'].min() < 0: # El R2 a veces es negativo si el modelo la lía mucho
                ax.set_ylim(df_filtrado['Valor'].min() * 1.1, y_max * 1.25)
            else:
                ax.set_ylim(0, y_max * 1.25)
                
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 3. Guardado
        # Limpio el nombre por si acso (ej: si tiene barras o espacios)
        nombre_limpio = nombre.replace('/', '_').replace(' ', '_').replace('.', '')
        ruta_guardado = carpeta_auditoria / f"Auditoria_{nombre_limpio}_{target_name}.png"
        plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
        plt.close() # Cierro para no petar la RAM
        
    print(f"   ✅ Auditorías generadas. Revisa la carpeta: {carpeta_auditoria.name}")

generar_auditoria_individual(mejores_modelos_p, X_train, y_train_p, X_test, y_test_p, "ISOPleasant")
generar_auditoria_individual(mejores_modelos_e, X_train, y_train_e, X_test, y_test_e, "ISOEventful")

# ==============================================================================
# 8.7. NUEVO: COMPARATIVA GLOBAL (TODAS LOS ML AGRUPADOS)
# ==============================================================================
def generar_comparativa_global_ias(df_resumen_global, target_name):
    """
    Cojo la tabla del ranking universal y monto un barplot agrupado
    para ver de un vistazo qué ML domina en qué métrica.
    """
    print(f"\n" + "🌟"*20)
    print(f" DIBUJANDO COMPARATIVA GLOBAL PARA: {target_name.upper()} ")
    print("🌟"*20)

    if df_resumen_global.empty:
        print(f"   ⚠️ No hay datos globales para {target_name}. Saltando gráfica.")
        return

    # 1. SELECCIÓN DE MÉTRICAS A COMPARAR
    # Cojo el MAE, RMSE y MSE (evalúan a nivel individuo)
    metricas_cols = ['MAE', 'RMSE', 'MSE']
    
    # Comprobación de seguridad por si he cambiado nombres arriba
    columnas_presentes = [col for col in metricas_cols if col in df_resumen_global.columns]
    
    if not columnas_presentes:
        print("   ⚠️ Faltan las columnas de métricas necesarias para la comparativa.")
        return

    # 2. TRANSFORMACIÓN DE DATOS (Melt)
    # Magia de Pandas: aplasta las columnas para que Seaborn pueda pintarlas juntas usando 'hue'
    df_melted = df_resumen_global.melt(
        id_vars='Modelo', 
        value_vars=columnas_presentes, 
        var_name='Métrica', 
        value_name='Valor'
    )
    
    # 3. DISEÑO DE LA GRÁFICA
    plt.figure(figsize=(15, 8))
    colores = ['#3498db', '#9b59b6', '#2ecc71'] 
    
    ax = sns.barplot(
        data=df_melted, x='Modelo', y='Valor', hue='Métrica', 
        palette=colores, edgecolor='black', linewidth=1.2
    )
    
    # Estética y Títulos
    plt.title(f'Rendimiento Global de los Algoritmos\nObjetivo: {target_name.upper()}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=15, ha='right', fontsize=12, fontweight='bold')
    plt.xlabel('Algoritmos de Machine Learning', fontsize=13, fontweight='bold', labelpad=15)
    plt.ylabel('Puntuación de la Métrica (Más bajo es MEJOR)', fontsize=13, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Métricas de Evaluación', loc='upper right', framealpha=0.9, fontsize=11)
    
    # Añadir números a las barras (los roto a 90 grados para que quepan si la barra es finita)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=5, fontsize=10, rotation=90)
        
    y_max = df_melted['Valor'].max()
    y_min = df_melted['Valor'].min()
    plt.ylim(y_min * 1.1 if y_min < 0 else 0, y_max * 1.4)

    plt.tight_layout()
    
    # Guardar la imagen
    carpeta_comparativas = ruta_script / 'Graficas_IA_AM2_TFG' / 'Comparativas_Globales'
    carpeta_comparativas.mkdir(parents=True, exist_ok=True)
    ruta_guardado = carpeta_comparativas / f"Comparativa_Global_IAs_{target_name}.png"
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Gráfica comparativa global guardada en: {ruta_guardado.name}")

# Llamamos a la comparativa pasándole los rankings que guardé en df_global_p y df_global_e
generar_comparativa_global_ias(df_global_p, "ISOPleasant")
generar_comparativa_global_ias(df_global_e, "ISOEventful")

# ==============================================================================
# 9. EXTRACCIÓN DE INTERPRETABILIDAD (PESOS Y TOP 15 VARIABLES)
# ==============================================================================
from sklearn.inspection import permutation_importance

def extraer_e_imprimir_importancias(mejores_modelos, X_train, y_train, target_name):
    """
    Abro "la cabeza" de cada ML para ver a qué variables acústicas le está haciendo caso.
    - Modelos Lineales -> Me dan los Coeficientes directamente.
    - Árboles -> Me dan Feature Importances (Gini).
    - Cajas Negras (SVM, GP) -> Uso Permutation Importance (mezclar una columna a ver si falla).
    """
    print(f"\n" + "="*50)
    print(f" EXTRAYENDO TOP 15 VARIABLES INDIVIDUALES: {target_name.upper()} ")
    print("="*50)
    
    carpeta_pesos = ruta_script / 'Graficas_IA_AM2_TFG' / 'Interpretabilidad'
    carpeta_pesos.mkdir(parents=True, exist_ok=True)
    
    nombres_variables = X_train.columns.tolist()
    
    for nombre_modelo, pipeline_optimo in mejores_modelos.items():
        print(f"   📊 Analizando los secretos de: {nombre_modelo}...")
        importancias = None
        es_coeficiente = False
        
        # Desmonto el pipeline temporalmente para ver al modelo puro y escalo la X a mano
        modelo_real = pipeline_optimo.named_steps['model']
        X_train_escalado = pipeline_optimo.named_steps['scaler'].transform(X_train)
        
        if nombre_modelo in ["RandomForest", "XGBoost"]:
            importancias = modelo_real.feature_importances_
            
        elif nombre_modelo in ["LinearRegression", "Lasso"]:
            importancias = modelo_real.coef_
            es_coeficiente = True
            
        else:
            print("      🧠 Aplicando Permutation Importance (barajando variables)...")
            # Truco de Scikit: barajo cada variable 5 veces a ver cómo impacta en el error (MAE)
            resultado_perm = permutation_importance(
                modelo_real, X_train_escalado, y_train, 
                n_repeats=5, random_state=42, scoring='neg_mean_absolute_error'
            )
            importancias = resultado_perm.importances_mean
            
        df_imp = pd.DataFrame({'Variable': nombres_variables, 'Valor': importancias})
        
        if es_coeficiente:
            # Los lineales pueden tener pesos negativos (ej: mucho ruido baja el agrado). 
            # Tomo el valor absoluto para rankear la fuerza bruta, sea a favor o en contra.
            df_imp['Fuerza'] = df_imp['Valor'].abs()
            df_imp = df_imp[df_imp['Fuerza'] > 0.0].sort_values(by='Fuerza', ascending=False).head(15)
        else:
            # Árboles y permutaciones ya dan valores absolutos de importancia
            df_imp = df_imp[df_imp['Valor'] > 0.001].sort_values(by='Valor', ascending=False).head(15)
            
        if len(df_imp) == 0: 
            print(f"      ⚠️ {nombre_modelo} ha anulado todas las variables. Saltando gráfica.")
            continue
            
        plt.figure(figsize=(10, 6))
        
        if es_coeficiente:
            # Rojo si resta puntuación, Verde si la suma
            colores = ["#e74c3c" if val < 0 else "#2ecc71" for val in df_imp['Valor']]
            ax = sns.barplot(data=df_imp, x='Valor', y='Variable', palette=colores, edgecolor='black')
            plt.xlabel("Peso del Coeficiente (Impacto direccional)")
            plt.title(f"Top 15 Factores Clave (Coeficientes) - {nombre_modelo}\n({target_name})")
        else:
            # Paleta de fueguito para la importancia genérica
            ax = sns.barplot(data=df_imp, x='Valor', y='Variable', palette="magma", edgecolor='black')
            plt.xlabel("Nivel de Importancia (Impacto en las decisiones de la IA)")
            plt.title(f"Top 15 Factores Clave (Importancia) - {nombre_modelo}\n({target_name})")
            
        plt.ylabel("Parámetros Acústicos")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Poner los valores exactos volando al final de cada barra
        for i, val in enumerate(df_imp['Valor']):
            texto = f"{val:.3f}" if es_coeficiente else f"{val:.4f}" 
            # abs() nativo de Python es vital aquí para que los tipos Float de numpy no crasheen
            offset = (df_imp['Valor'].max() * 0.02) if val >= 0 else -(abs(df_imp['Valor'].min()) * 0.02)
            ha_align = 'left' if val >= 0 else 'right'
            ax.text(val + offset, i, texto, va='center', ha=ha_align, fontsize=10, fontweight='bold')
            
        # Alargar un poco el eje X para que el texto de las barras nunca se quede fuera de plano
        x_min, x_max = plt.xlim()
        rango = x_max - x_min
        plt.xlim(x_min - (rango*0.1) if x_min < 0 else 0, x_max + (rango*0.15))
        
        plt.tight_layout()
        
        # Guardo el top 15
        nombre_limpio = nombre_modelo.replace('/', '_').replace(' ', '_').replace('.', '')
        ruta_img = carpeta_pesos / f"Pesos_{nombre_limpio}_{target_name}.png"
        plt.savefig(ruta_img, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      ✅ Gráfica generada: {ruta_img.name}")

# Llamamos a la función. A las "cajas negras" les hace falta la 'y_train' para permutar
extraer_e_imprimir_importancias(mejores_modelos_p, X_train, y_train_p, "ISOPleasant")
extraer_e_imprimir_importancias(mejores_modelos_e, X_train, y_train_e, "ISOEventful")

# ==============================================================================
# 10. ENSAMBLAJE (ENVERGUDURA GLOBAL): LA "SÚPER PREDICCIÓN"
# ==============================================================================
def generar_super_prediccion_ensamblada(mejores_modelos, df_ranking_universal, X_test, y_test_real, target_name):
    """
    Combino "lo mejor de los dos mundos": el mejor modelo simple/lineal con el 
    mejor modelo complejo. Sus errores suelen compensarse al hacer la media.
    """
    print("\n" + "🌟"*25)
    print(f" GENERANDO SÚPER PREDICCIÓN ENSAMBLADA: {target_name.upper()} ")
    print("🌟"*25)
    
    if df_ranking_universal.empty:
        return
        
    lineales = ["LinearRegression", "Lasso"]
    complejos = ["RandomForest", "XGBoost", "SVR", "GaussianProcess"]
    
    # Filtro mi ranking global para buscar a los campeones de cada categoría
    df_lineales = df_ranking_universal[df_ranking_universal['Modelo'].isin(lineales)]
    df_complejos = df_ranking_universal[df_ranking_universal['Modelo'].isin(complejos)]
    
    if df_lineales.empty or df_complejos.empty: return
        
    mejor_lineal = df_lineales.sort_values(by='MAE (Error Absoluto)').iloc[0]['Modelo']
    mejor_complejo = df_complejos.sort_values(by='MAE (Error Absoluto)').iloc[0]['Modelo']
    
    print(f"   🏆 Campeón Lineal: {mejor_lineal} | Campeón Complejo: {mejor_complejo}")
    
    # Extraigo la predicción de cada campeón
    pred_lineal = mejores_modelos[mejor_lineal].predict(X_test)
    pred_compleja = mejores_modelos[mejor_complejo].predict(X_test)
    
    # Calculo la Súper Predicción (Ensamblado Promediado / Soft-Voting)
    super_prediccion = (pred_lineal + pred_compleja) / 2.0
    
    # Evalúo si este "Frankenstein" ha superado a sus modelos padre
    mae_super = mean_absolute_error(y_test_real, super_prediccion)
    rmse_super = np.sqrt(mean_squared_error(y_test_real, super_prediccion))
    r2_super = r2_score(y_test_real, super_prediccion)
    
    print(f"   📊 SÚPER MODELO -> R2: {r2_super:.4f} | MAE: {mae_super:.4f} | RMSE: {rmse_super:.4f}")
    
    # Dibujo la matriz de dispersión (Realidad vs ML)
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test_real, y=super_prediccion, alpha=0.6, color='#8e44ad', s=60, edgecolor='white')
    
    # Trazo la diagonal roja: donde los puntos deberían caer si mi ML fuera adivina al 100%
    limite_min = min(y_test_real.min(), super_prediccion.min()) - 0.1
    limite_max = max(y_test_real.max(), super_prediccion.max()) + 0.1
    plt.plot([limite_min, limite_max], [limite_min, limite_max], color='red', linestyle='--', linewidth=2)
    
    plt.title(f'Realidad vs Súper Predicción\n({mejor_lineal} + {mejor_complejo}) - {target_name}', fontsize=14, pad=15)
    plt.xlabel('Respuesta Real (Groningen)')
    plt.ylabel('Súper Predicción Ensamblada')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Pongo un cuadrito con la nota final (R2 y MAE) arriba a la izquierda
    texto_box = f"$R^2$ = {r2_super:.4f}\nMAE = {mae_super:.4f}"
    plt.text(0.05, 0.95, texto_box, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
             
    plt.tight_layout()
    carpeta_ensamble = ruta_script / 'Graficas_IA_AM2_TFG' / 'Ensamblaje'
    carpeta_ensamble.mkdir(parents=True, exist_ok=True)
    plt.savefig(carpeta_ensamble / f"Super_Prediccion_{target_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

# Disparo las funciones pasándole el dataframe del ranking universal que creamos antes
generar_super_prediccion_ensamblada(mejores_modelos_p, df_global_p, X_test, y_test_p, "ISOPleasant")
generar_super_prediccion_ensamblada(mejores_modelos_e, df_global_e, X_test, y_test_e, "ISOEventful")

print("\n" + "🚀"*20)
print(" PIPELINE DE MACHINE LEARNING FINALIZADO COMPLETAMENTE ")
print("🚀"*20)