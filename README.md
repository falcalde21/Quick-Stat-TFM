# Quick, Stat!: Un análisis completo de la base de datos Quick, Draw! usando redes recurrentes

Este repositorio contiene el modelo y los pesos del clasificador RNN utilizado en el TFM. El modelo clasifica las 345 categorías del dataset de QuickDraw en formato TFRecord con una precisión del 76.8%.

## Contenidos

- `model_weights-0.768.h5`: Pesos del modelo entrenado.
- `create_TFRecord_datasets.py`: Código utilizado para convertir el dataset de QuickDraw a formato TFRecord, creando conjuntos de datos de entrenamiento y validación.
- `train_RNN.py`: Código utilizado para entrenar el modelo RNN.

