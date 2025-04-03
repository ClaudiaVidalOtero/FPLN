# Módulos estándar de Python
import os
import re
from typing import Dict, List

# Librerías de terceros
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# TensorFlow y Keras
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.layers import (Input, Embedding, Lambda, Dense, Dropout, BatchNormalization, LSTM, Bidirectional)
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping


def build_model(vocab_size, embedding_dim, window_size):
    
    # 1. Capa de Entrada
    input_context = layers.Input(shape=(window_size-1,), name='context_input')

    # 2. Capa Embedding
    embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='embedding')(input_context)

    # 3. Capa de Promedio (usando Lambda)
    average_layer = layers.Lambda(lambda x: K.mean(x, axis=1), name='average_embedding')(embedding_layer)

    # 4. Capa Densa de Salida + Softmax
    output_layer = layers.Dense(units=vocab_size, activation='softmax', name='output_dense_softmax')(average_layer)

    # --- Creación del Modelo Keras ---
    model = keras.Model(inputs=input_context, outputs=output_layer, name='Modelo_CBOW_1')

    # --- Compilación del Modelo ---
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model



def generar_muestras(token_ids, window_size):
    context_half_size = (window_size - 1) // 2
    muestras = [(token_ids[i - context_half_size:i] + token_ids[i + 1:i + 1 + context_half_size], token_ids[i])
                for i in range(context_half_size, len(token_ids) - context_half_size)]
    return muestras


def obtener_contextos_pretokenizados(token_ids, palabras_objetivo, window_size, word_index, index_word):
    
    # En caso de que la palabra no esté en el índice
    if not token_ids:
        print("Advertencia: La secuencia de token_ids está vacía.")
        return {palabra.lower().strip(): [] for palabra in palabras_objetivo if palabra.strip()}

    # Generamos muestras del contexto 
    try:
        muestras_enteras = generar_muestras(token_ids, window_size)
    except ValueError as e:
        print(f"Error al generar muestras: {e}")
        return {palabra.lower().strip(): [] for palabra in palabras_objetivo if palabra.strip()}

    # Buscamos el id de cada palabra pedida
    resultados = {}
    target_ids_map = {word_index[palabra.lower().strip()]: palabra.lower().strip()
                      for palabra in palabras_objetivo if palabra.lower().strip() in word_index}
    
    
    for palabra in target_ids_map.values():
        resultados[palabra] = []

    #Busca las palabras en los contextos
    total_contextos_encontrados = 0
    for contexto_ids, objetivo_id in muestras_enteras:
        # Si el id de la muestra coincide con uno de los id buscados
        if objetivo_id in target_ids_map:
            palabra_encontrada = target_ids_map[objetivo_id]
            contexto_palabras = [index_word.get(cid, f"<ID_{cid}?>") for cid in contexto_ids]
            resultados[palabra_encontrada].append(contexto_palabras)
            total_contextos_encontrados += 1

    # En caso de no encontrar las palabras objetivo
    if total_contextos_encontrados == 0:
        print(f"\nAdvertencia: Ninguna de las palabras objetivo ({list(target_ids_map.values())}) fue encontrada como palabra central.")
    #En caso de no encontrar contextos para las palabras 
    else:
        palabras_no_encontradas = set(target_ids_map.values()) - set(resultados.keys())
        if palabras_no_encontradas:
            print(f"\nAdvertencia: No se encontraron contextos para: {list(palabras_no_encontradas)}")

    return resultados


from collections import defaultdict
import random

def balancear_muestras(muestras, max_muestras_por_palabra=1000):
    frecuencia = defaultdict(list)

    # Organizar muestras por palabra objetivo
    for contexto, palabra in muestras:
        frecuencia[palabra].append((contexto, palabra))

    # Aplicar límite a palabras frecuentes
    muestras_balanceadas = []
    for palabra, lista in frecuencia.items():
        if len(lista) > max_muestras_por_palabra:
            muestras_balanceadas.extend(random.sample(lista, max_muestras_por_palabra))  # Limita aleatoriamente
        else:
            muestras_balanceadas.extend(lista)

    return muestras_balanceadas


def preparar_datos(token_ids, vocab_size, window_size, test_size=0.2, dev_size=0.1):
    muestras = generar_muestras(token_ids, window_size)
    train_muestras, test_muestras = train_test_split(muestras, test_size=test_size + dev_size, random_state=42)
    train_muestras, dev_muestras = train_test_split(train_muestras, test_size=dev_size / (test_size + dev_size), random_state=42)
    
    train_muestras = balancear_muestras(train_muestras, max_muestras_por_palabra=20)

    def convertir_muestras(muestras):
        X = np.array([contexto for contexto, _ in muestras])  # Contexto
        y = np.array([objetivo for _, objetivo in muestras])  # Palabra objetivo (como número entero)
        return X, y  # Sin one-hot encoding

    
    return convertir_muestras(train_muestras), convertir_muestras(dev_muestras), convertir_muestras(test_muestras)



def entrenar_modelo(token_ids, vocab_size, window_size=5, embedding_dim=50, epochs=10, batch_size=128):
    
    (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = preparar_datos(token_ids, vocab_size, window_size)

     # 2. Construir Modelo
    model = build_model(vocab_size, embedding_dim, window_size)
    model.summary() 

    early_stop = EarlyStopping(
        monitor='val_loss',      
        patience=10,             
        restore_best_weights=True, 
        verbose=1                
    )

    # 4. Entrenar Modelo
    print(f"\nIniciando entrenamiento por hasta {epochs} épocas...")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_dev, y_dev), # Usa el conjunto de validación
        callbacks=[early_stop],         # Aplica el early stopping
        verbose=1                       # Muestra progreso (1 para barra, 2 para línea por epoch)
    )

    # 5. Evaluar Modelo
    print("\nEvaluando el modelo en el conjunto de prueba...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Rendimiento en la prueba: Pérdida = {test_loss:.4f}, Precisión = {test_acc:.4f}")

    # 6. Extraer Pesos de Embedding (Corrección aquí)
    print("Extrayendo pesos de la capa de embedding...")
    # Usa el nombre correcto 'embedding' que definiste en build_model
    embedding_weights = model.get_layer("embedding").get_weights()[0]
    print(f"Forma de la matriz de embeddings: {embedding_weights.shape}") # Debería ser (vocab_size, embedding_dim)

    return model, X_test, y_test, embedding_weights



def calcular_similitud(embedding_layer, word_index, index_word, palabras_objetivo, top_n=10):
    resultados = {}
    for palabra in palabras_objetivo:
        if palabra in word_index:
            palabra_id = word_index[palabra]
            vector = embedding_layer[palabra_id].reshape(1, -1)
            similitudes = cosine_similarity(vector, embedding_layer)[0]
            palabras_mas_similares = np.argsort(similitudes)[::-1][1:top_n + 1]
            resultados[palabra] = [index_word.get(idx, f"<ID_{idx}?>") for idx in palabras_mas_similares]
    return resultados


def visualizar_tsne(embedding_layer, word_index, palabras_objetivo, titulo="Embeddings con t-SNE"):
    ids = [word_index[palabra] for palabra in palabras_objetivo if palabra in word_index]
    vectores = np.array([embedding_layer[i] for i in ids])
    palabras = [palabra for palabra in palabras_objetivo if palabra in word_index]
    tsne = TSNE(n_components=2, random_state=42)
    vectores_2d = tsne.fit_transform(vectores)
    plt.figure(figsize=(10, 6))
    for i, palabra in enumerate(palabras):
        plt.scatter(vectores_2d[i, 0], vectores_2d[i, 1], label=palabra)
        plt.text(vectores_2d[i, 0], vectores_2d[i, 1], palabra, fontsize=12)
    plt.legend()
    plt.title(titulo)
    plt.show()




# Cargamos el texto de Harry Potter
ruta_archivo = os.path.join("harry_potter_and_the_philosophers_stone.txt")

with open(ruta_archivo, 'r', encoding='utf-8') as file:
    texto_harry_potter = file.read()
print(f"Archivo '{ruta_archivo}' leído.")


print("\n--- Tokenizando el texto ---")

# Tokenizamos el texto
tokenizer_hp = Tokenizer()
tokenizer_hp.fit_on_texts([texto_harry_potter])
sequences = tokenizer_hp.texts_to_sequences([texto_harry_potter])
token_ids_hp = sequences[0] if sequences else []
word_index_hp = tokenizer_hp.word_index
index_word_hp = tokenizer_hp.index_word
vocab_size_hp = len(word_index_hp) + 1

print(f"Tokenización completada.")
print(f"  - Tamaño del vocabulario: {vocab_size_hp - 1} palabras únicas.")
print(f"  - Longitud de la secuencia de IDs: {len(token_ids_hp)}")


# Definir parámetros del modelo
window_size = 5  # 2 palabras antes y 2 después de la palabra objetivo
embedding_dim = 100
epochs = 50
batch_size = 64

# Generar muestras de entrenamiento
(X_train, y_train), (X_dev, y_dev), (X_test, y_test) = preparar_datos(token_ids_hp, vocab_size_hp, window_size)

# Construir y entrenar el modelo
modelo, X_test, y_test, embedding_layer = entrenar_modelo(token_ids_hp, vocab_size_hp, window_size, embedding_dim, epochs, batch_size)



# Palabras objetivo para visualizar contextos y similitudes
palabras_objetivo = ["harry", "hogwarts", "magic", "wand", "wizard"]

# Obtener contextos para palabras objetivo
contextos = obtener_contextos_pretokenizados(token_ids_hp, palabras_objetivo, window_size, word_index_hp, index_word_hp)
print("\nContextos encontrados:")
for palabra, contexto in contextos.items():
    print(f"{palabra}: {contexto[:3]} ...")  # Muestra solo los primeros 3 contextos
