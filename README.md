# Spotify Tracks: predicción de popularidad de canciones

Proyecto de la asignatura **Machine Learning (MLY1101)**, Evaluación Parcial N°1, Caso C.

**Integrantes:** Moira Bittner, Lucas Fernandez y Julian Martinez

En este proyecto analizamos el dataset *Spotify Tracks* (114.000 canciones) para identificar qué factores se relacionan con la popularidad de una canción y preparar los datos para un modelo de regresión. Seguimos la metodología CRISP-DM hasta la fase de preparación de los datos.

📄 **El desarrollo completo está en el [Informe técnico](Informe_Tecnico_EP1_Spotify.md).**

## Estructura del proyecto

```
spotify-popularidad-ml/
├── README.md
├── Informe_Tecnico_EP1_Spotify.md   # informe técnico del proyecto
├── requirements.txt                 # librerías necesarias
├── data/
│   ├── raw/                         # dataset original
│   └── processed/                   # dataset limpio (generado por el notebook)
├── notebooks/
│   └── EP1_Spotify_Moira_Lucas_Julian.ipynb
├── images/                          # gráficos usados en el informe
└── models/                          # modelos entrenados (próxima etapa)
```

## Cómo ejecutar

### Opción 1: en el computador

```bash
pip install -r requirements.txt
jupyter notebook notebooks/EP1_Spotify_Moira_Lucas_Julian.ipynb
```

Luego ejecutar todas las celdas en orden.

### Opción 2: en Google Colab

1. Subir la carpeta completa `spotify-popularidad-ml` a Google Drive (en *Mi unidad*).
2. Abrir el notebook desde Drive con Google Colab.
3. Agregar y ejecutar esta celda al inicio, antes de las demás:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/spotify-popularidad-ml/notebooks
```

4. Ejecutar el resto de las celdas en orden.

## Resultados generados

Al ejecutar el notebook se generan:

- `data/processed/Spotify_Tracks_Dataset_Limpio.csv`: dataset limpio (113.549 registros).
- `images/`: gráficos de distribución de la popularidad, matriz de correlación y popularidad por género.

Todas las operaciones aleatorias usan la semilla `42`, por lo que los resultados son reproducibles.
