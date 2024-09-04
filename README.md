# Quick, Stat!: Un análisis completo de la base de datos Quick, Draw! usando redes recurrentes

Este repositorio contiene el modelo y los pesos del clasificador RNN utilizado en el TFM. Es un modelo implementado en Keras utilizando TensorFlow 2. El modelo clasifica las 345 categorías del dataset de QuickDraw en formato TFRecord con una precisión del 76.8%.

## Contenidos

- `model_weights-0.768.h5`: Pesos del modelo entrenado.
- `create_TFRecord_datasets.py`: Código utilizado para convertir el dataset de QuickDraw a formato TFRecord, creando conjuntos de datos de entrenamiento y validación.
- `train_RNN.py`: Código utilizado para entrenar el modelo RNN.

## Definición del Modelo

El siguiente código muestra la función `build_model` que define la arquitectura del modelo RNN:

```python
from tensorflow.keras import layers, models

def build_model():
    model = models.Sequential()
    # Capa de entrada y filtro
    model.add(layers.Input(shape=(3711, 3)))
    model.add(layers.Masking(mask_value=0.0))
    # Capas convolucionales
    model.add(layers.Conv1D(48, 5, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(64, 5, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(96, 3, activation='relu'))
    model.add(layers.BatchNormalization())
    # LSTM
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
    # Pooling y Softmax
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(345, activation='softmax'))
    return model
```

## Cargar el Modelo Preentrenado

Para utilizar el modelo ya entrenado se pueden cargar los pesos utilizando el siguiente código:

```python
# Cargar el modelo y los pesos
MODEL_WEIGHTS_PATH = 'model_weights-0.768.h5'
trained_model = build_model()
trained_model.load_weights(MODEL_WEIGHTS_PATH)
