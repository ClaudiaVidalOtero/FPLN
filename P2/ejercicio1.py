import os
import re
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def build_model(vocab_size, embedding_dim, window_size):

    input_context = layers.Input(shape=(window_size-1,), name='context_input')

    embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='embedding')(input_context)
    average_layer = layers.Lambda(lambda x: K.mean(x, axis=1), name='average_embedding')(embedding_layer)
    
    output_layer = layers.Dense(units=vocab_size, activation='softmax', name='output_dense_softmax')(average_layer)
    
    model = keras.Model(inputs=input_context, outputs=output_layer, name='Modelo_CBOW_1')
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


def generar_muestras(token_ids, window_size):
    context_half_size = (window_size - 1) // 2
    return [(token_ids[i - context_half_size:i] + token_ids[i + 1:i + 1 + context_half_size], token_ids[i])
            for i in range(context_half_size, len(token_ids) - context_half_size)]


def balancear_muestras(muestras, max_muestras_por_palabra=1000):
    frecuencia = defaultdict(list)
    for contexto, palabra in muestras:
        frecuencia[palabra].append((contexto, palabra))
    muestras_balanceadas = []
    for palabra, lista in frecuencia.items():
        if len(lista) > max_muestras_por_palabra:
            muestras_balanceadas.extend(random.sample(lista, max_muestras_por_palabra))
        else:
            muestras_balanceadas.extend(lista)
    return muestras_balanceadas


def preparar_datos(token_ids, vocab_size, window_size):
    muestras = generar_muestras(token_ids, window_size)
    
    def convertir_muestras(muestras):
        X = np.array([contexto for contexto, _ in muestras])
        y = np.array([objetivo for _, objetivo in muestras])
        return X, y

    return convertir_muestras(muestras)




def entrenar_modelo(token_ids, vocab_size, window_size=5, embedding_dim=100, epochs=20, batch_size=128):
    (X, y) = preparar_datos(token_ids, vocab_size, window_size)

    # División en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    model = build_model(vocab_size, embedding_dim, window_size)
    model.summary()
    embedding_inicial = model.get_layer("embedding").get_weights()[0]
    visualizar_tsne(embedding_inicial, word_index_hp, palabras_objetivo, titulo="Embeddings Iniciales con t-SNE")

    print(f"\nIniciando entrenamiento por hasta {epochs} épocas...")

    # Callback de early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    # Obtener la matriz de embeddings
    embedding_weights = model.get_layer("embedding").get_weights()[0]
    print(f"Forma de la matriz de embeddings: {embedding_weights.shape}")

    return model, X_train, y_train, embedding_weights

def calcular_similitud(embedding_layer, word_index, index_word, palabras_objetivo, top_n=10, mostrar=True):
    resultados = {}
    for palabra in palabras_objetivo:
        if palabra in word_index:
            palabra_id = word_index[palabra]
            vector = embedding_layer[palabra_id].reshape(1, -1)
            similitudes = cosine_similarity(vector, embedding_layer)[0]
            palabras_mas_similares = np.argsort(similitudes)[::-1][1:top_n + 1]
            similares = [index_word.get(idx, f"<ID_{idx}?>") for idx in palabras_mas_similares]
            resultados[palabra] = similares
            if mostrar:
                print(f"{palabra}: {', '.join(similares)}")
        else:
            if mostrar:
                print(f"{palabra}: [Palabra no encontrada en el vocabulario]")
    return resultados



def visualizar_tsne(embedding_layer, word_index, palabras_objetivo, titulo="Embeddings con t-SNE"):
    ids = [word_index[palabra] for palabra in palabras_objetivo if palabra in word_index]
    if len(ids) < 2:
        print("No hay suficientes palabras para aplicar t-SNE")
        return
    
    vectores = np.array([embedding_layer[i] for i in ids])
    palabras = [palabra for palabra in palabras_objetivo if palabra in word_index]
    
    tsne = TSNE(n_components=2, perplexity=min(5, len(vectores)-1), random_state=42)
    vectores_2d = tsne.fit_transform(vectores)
    
    plt.figure(figsize=(10, 6))
    for i, palabra in enumerate(palabras):
        plt.scatter(vectores_2d[i, 0], vectores_2d[i, 1], label=palabra)
        plt.text(vectores_2d[i, 0], vectores_2d[i, 1], palabra, fontsize=12)
    plt.title(titulo)
    plt.show()


nombre_archivo = os.path.join("target_words_harry_potter.txt")
def leer_palabras_desde_archivo(nombre_archivo):
    with open(nombre_archivo, "r", encoding="utf-8") as archivo:
        palabras = [linea.strip() for linea in archivo if linea.strip()]
    return palabras

palabras_objetivo = leer_palabras_desde_archivo(nombre_archivo)
print(palabras_objetivo)


ruta_archivo = os.path.join("harry_potter_and_the_philosophers_stone.txt")
with open(ruta_archivo, 'r', encoding='utf-8') as file:
    texto_harry_potter = file.read()
print(f"Archivo '{ruta_archivo}' leído.")

tokenizer_hp = Tokenizer()
tokenizer_hp.fit_on_texts([texto_harry_potter])
sequences = tokenizer_hp.texts_to_sequences([texto_harry_potter])
token_ids_hp = sequences[0] if sequences else []
word_index_hp = tokenizer_hp.word_index
index_word_hp = tokenizer_hp.index_word
vocab_size_hp = len(word_index_hp) + 1

window_size = 5
embedding_dim = 100
epochs = 30
batch_size = 128
modelo, X_test, y_test, embedding_layer = entrenar_modelo(token_ids_hp, vocab_size_hp, window_size, embedding_dim, epochs, batch_size)


visualizar_tsne(embedding_layer, word_index_hp, palabras_objetivo)


print("SIMILITUDES DE PALABRAS OBJETIVO:\n")

similares_por_palabra = calcular_similitud(embedding_layer, word_index_hp, index_word_hp, palabras_objetivo)
