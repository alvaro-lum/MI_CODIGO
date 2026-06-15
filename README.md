# Guía de Configuración del Entorno de Trabajo

Este repositorio contiene el código desarrollado para el Trabajo Fin de Grado (TFG). A continuación, se detalla la estructura de archivos requerida y los pasos necesarios para configurar el entorno virtual de ejecución.

> ℹ️ **Nota:** Este proyecto ha sido desarrollado utilizando **VS Code**, pero puedes emplear el editor de código o IDE de tu preferencia.

---

## 📂 Estructura del Proyecto

Para garantizar el correcto funcionamiento de los scripts, es fundamental mantener la siguiente estructura de carpetas. Los códigos contienen rutas relativas y absolutas que apuntan a estos directorios; **si modificas esta distribución, los scripts darán error de ruta** a menos que los actualices manualmente. Si la modificas o empleas otro tipo de distribución no olvides de modificar dichas rutas para evitar errores.

La distribución recomendada dentro de la carpeta raíz (`MI_TFG`) es:

```text
MI_TFG/
├── DATOS/               # Material extraído del repositorio de SSID.
├── MI_CODIGO/           # Archivos y scripts de este repositorio de GitHub.
└── soundscape_env/      # Entorno virtual de Python (se creará a continuación).
```

## 🛠️ Configuración del Entorno Virtual

Sigue estos pasos para crear y activar el entorno virtual donde se ejecutarán los scripts:

1. Abre tu editor de código con la carpeta raíz del proyecto (MI_TFG).

2. Abre una terminal y asegúrate de estar en la ruta de esa carpeta madre.

3. Crea el entorno virtual ejecutando el siguiente comando:
```python
python -m venv soundscape_env
```
4. Activa el entorno virtual según tu sistema operativo:

Windows:
```python
soundscape_env\Scripts\activate
```
Mac / Linux:

```python
source soundscape_env/bin/activate
```
📦 Instalación de Dependencias

Dispones de dos opciones para instalar las librerías necesarias.

Opción 1: Instalación rápida (Recomendada)
Si ya has descargado el archivo requirements.txt de este repositorio, puedes replicar exactamente mi configuración ejecutando:

```python
pip install -r requirements.txt
```
Opción 2: Instalación manual paso a paso
Si prefieres instalar los paquetes manualmente, ejecuta los siguientes comandos en orden:

Instalar el pipeline principal de datos y modelado:

```python
pip install pandas numpy scipy scikit-learn xgboost matplotlib seaborn
```
Instalar las utilidades restantes (procesamiento de audio e interfaces):

```python
pip install librosa chameleon-agents chameleon-ui
```
(Nota: Este entorno se consolidó ejecutando el comando pip freeze > requirements.txt).

🔍 Solución de Problemas: Reactivación en VS Code

Si cierras el programa y, al volverlo a abrir, notas que no te encuentras dentro del entorno virtual (no aparece el prefijo (soundscape_env) en la terminal), sigue estos pasos desde la interfaz de VS Code:

1. Pulsa la combinación de teclas Ctrl + Shift + P (o F1) para abrir la paleta de comandos.

2. Escribe Python: Select Interpreter (Seleccionar intérprete) y pulsa Enter.

En la lista desplegable que aparece, busca y selecciona la opción que corresponda a tu entorno. Debería lucir similar a esto:

```Plaintext
Python 3.1x.x ('soundscape_env': venv) .\soundscape_env\Scripts\python.exe
```
3. Una vez seleccionado, cierra la terminal actual (haz clic en el icono de la papelera en la esquina de la ventana de la terminal).

4. Abre una nueva terminal desde el menú superior: Terminal -> New Terminal.

Si la configuración es correcta, verás el texto (soundscape_env) al principio de la línea de comandos.

⚠️ Recuerda: El proceso de selección del intérprete solo se realiza una vez. Sin embargo, cada vez que abras de nuevo VS Code, es recomendable cerrar la terminal antigua que se abre por defecto y abrir una nueva para asegurarte de que el entorno se cargue correctamente.

## Configuración de carpetas y códigos.

Una vez establecido el entorno de trabajo debo explicar a que carpeta corresponde cada código aqui presente:
(Carpeta --> Código)

1. graficas_TFG --> analisis.py
2. Graficas_Londres_TFG --> analisis_londres.py
3. Graficas_Groningen_TFG --> analisis_groningen.py
4. Graficas_Granada_TFG --> analisis_granada.py
5. Graficas_Venecia_TFG --> analisis_venice.py
6. Graficas_Audios_TFG --> analisis_audios.py
7. Repetidas --> Prueba_repetidas.py
8. Graficas_IA_TFG --> aprendizaje_maquina.py
9. Graficas_IA_M2_TFG --> aprendizaje_maquina_2.py
10. Graficas_IA_AM1_TFG --> AM_Global_1.py
11. Graficas_IA_AM2_TFG --> AM_Global_2.py
12. Graficas_IA_Ldn1_TFG --> AM_Londres_1.py
13. Graficas_IA_Ldn2_TFG --> AM_Londres_2.py
14. Graficas_IA_COMBINADO_TFG y Graficas_IA_SUSTITUIDO_TFG --> Prueba_Londres_ia_1.py
15. Prueba_TFG --> ia_modelos.py

## Como funcionan los códigos en terminal.

Para poder inicir a correr un código deberemos clicar en la flecha que encontramos en VS Code alrriba a la derecha, se abrira un terminal (si no esta abierto) y se ejecutará.

Destacamos que los códigos que van de los numeros 1 al 6 del bloque anterior, al ejecutarlos en la terminal podremos observar un índice creado por mi para que sea más fácil su utilización, deberemos seguir las indicaciones que nos pide para crear los archivos que queramos en cada código.

El número 7 es solo una prueba realizada y desmentida que se realizó en el transcurso del trabajo (No es importante). Al igual que el número 15, que solo son pruebas.

Y los número entre 8 y 14, ambos incluidos, cuando los ejecutamos estos se generan solos, salvo el 14, que saldrá otro índice para decirnos que opción preferimos realizar. 

> ⚠️ **Aviso:** Los códigos del 8 al 14 tardan en ejecutarse al tener que realizar todo el proceso de **GrideSearch** y demás (Tiempo aproximado entre 10-15 min, posible que sean 20 min). Y el código número 6, el de audio, tiene una opción, la última que encontrarás, que al ejecutarla tardará mucho tiempo en completarse al juntar muchos datos de tantos audios (En mi caso fueron aproximadamente 5h). 

### Algunos extras.

Puedes comprobar que además de todo lo explicado con anterioridad, podemos encontrar el archivo requirements.txt que ya expliqué y el documento ISD v1.0 Data.csv. Este archivo es la base de datos general extraida del repositorio de SSID y del que partimos en todos los códigos del 1 al 6.

> ℹ️ **Nota:** Ese csv se encontraría en la carpeta DATOS, ya que los códigos la cargan desde ahí. Está en esta carpeta para subirla a este repositorio ya que es de uso libre, como todo lo del repositorio de SSID, pero acuerdate de situar este archivo en dicha carpeta o cambiar la ruta de cada código para que lo lea correctamente.

