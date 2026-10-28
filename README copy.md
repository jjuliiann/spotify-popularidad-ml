# Inteligencia musical y predicción de popularidad de canciones (Spotify Tracks)

Proyecto de Machine Learning de la asignatura **Machine Learning (MLY1101)**, caso C - Spotify Tracks.  
**Integrantes:** Moira Bittner, Lucas Fernandez y Julian Martinez.

Un sello discográfico necesita decidir en qué canciones invertir su presupuesto de promoción. Este proyecto estima la **popularidad** de una canción (0 a 100) a partir de su género y sus características de audio, para priorizar las de mayor potencial, y describe el catálogo mediante **perfiles de sonido**.

## Resultados principales

| | Resultado |
|---|---|
| Modelo final | **Gradient Boosting** (`HistGradientBoostingRegressor`) |
| Error promedio (MAE) | **9,06 puntos**, frente a 9,88 de la regla "promedio del género" y 15,19 de predecir siempre el promedio |
| Precisión en el top 10 % | **80,5 %** de canciones de alto potencial (popularidad ≥ 50), frente a 75,6 % de la regla y 26,5 % al azar |
| R² ajustado | 0,486, frente a 0,431 de la regla |
| Modelos comparados | Ridge, Random Forest y Gradient Boosting |
| Aprendizaje no supervisado | K-Means (4 perfiles de sonido), PCA y agrupamiento jerárquico de Ward para validar los perfiles |

El detalle completo está en los informes técnicos:

- [`EP1-ML-INFORME.md`](EP1-ML-INFORME.md): problema de negocio, métricas, diccionario de datos, EDA, calidad de datos, preparación y sesgos.
- [`EP2-ML-INFORME.md`](EP2-ML-INFORME.md): modelos supervisados y no supervisados, evaluación, selección del modelo y recomendaciones.

## Estructura del proyecto

```text
spotify-popularidad-ml/
├── README.md
├── EP1-ML-INFORME.md                     # informe técnico EP1
├── EP2-ML-INFORME.md                     # informe técnico EP2
├── requirements.txt
├── data/
│   ├── raw/Spotify_Tracks_Dataset.csv    # dataset original (Kaggle)
│   └── processed/Spotify_Tracks_Dataset_Limpio.csv   # generado por el notebook EP1
├── notebooks/
│   ├── EP1-ML.ipynb                      # EDA, limpieza y preparación
│   └── EP2_ML.ipynb                      # modelamiento y evaluación
├── images/                               # gráficos generados por los notebooks (prefijo ep2_ para la EP2)
└── models/
    ├── escalador_standard.joblib         # EP1: escalador de las variables continuas
    ├── modelo_final.joblib               # EP2: Gradient Boosting con su preprocesamiento
    └── kmeans_perfiles.joblib            # EP2: escalador + K-Means de perfiles de sonido
```

## Cómo ejecutar

### En un computador local

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Ejecutar todas las celdas de `EP1-ML.ipynb` y luego de `EP2_Spotify_Moira_Lucas_Julian.ipynb`. Los notebooks leen el dataset desde `data/raw/`, y guardan los gráficos en `images/` y los modelos en `models/`. La EP2 no depende de los archivos que genera la EP1: reproduce la misma preparación desde el CSV original.

### En Google Colab

1. Abrir el notebook en Colab (las librerías ya vienen instaladas).
2. Ejecutar todas las celdas en orden. Si el notebook no encuentra `data/raw/`, la celda de carga pide subir `Spotify_Tracks_Dataset.csv`.
3. La EP1 tarda menos de un minuto y la EP2 entre 10 y 15 minutos; el paso más lento es Random Forest. Los gráficos y modelos se guardan en las carpetas `images/` y `models/` del entorno de Colab.

## Reproducibilidad

- Todas las operaciones aleatorias usan la **semilla 42**.
- Las cifras de los informes se obtuvieron con Python 3.12, pandas 3.0.2, NumPy 2.4.4 y scikit-learn 1.8.0 (`requirements.txt`).
- `HistGradientBoostingRegressor` puede variar en centésimas entre equipos o versiones de scikit-learn. Esas diferencias no cambian el orden de los modelos ni las conclusiones.
- Los archivos `.joblib` dependen de la versión de scikit-learn con que se guardaron. Si al cargarlos aparece un error de versión, basta con ejecutar el notebook correspondiente para volver a generarlos.

## Usar el modelo final

```python
import joblib
import numpy as np

modelo = joblib.load("models/modelo_final.joblib")
# cancion: DataFrame con las 15 variables de entrada:
# danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness,
# valence, tempo, duration_log (= np.log(duración en minutos)), explicit, mode, key,
# time_signature, track_genre
popularidad_estimada = modelo.predict(cancion)
```

El `Pipeline` incluye el preprocesamiento, así que no es necesario escalar ni codificar los datos a mano.

## Fuente de datos y uso

- Pandya, M. *Spotify Tracks Dataset*. Kaggle: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
- Los datos provienen de la API de Spotify, cuyos términos para desarrolladores restringen el uso de su contenido para entrenar modelos de Machine Learning. Este proyecto tiene **fines exclusivamente académicos**.
