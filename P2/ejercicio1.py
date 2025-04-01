
import os
import re
from typing import Dict, List
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Embedding, Input, Lambda 
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def build_model(vocab_size=10000, embedding_dim=100, context_size=4):
    
    # Capa de entrada
    input_context = Input(shape=(context_size,), name='context_input')
    
    # Capa de Embedding
    embedding_layer = Embedding(input_dim=vocab_size, 
                                output_dim=embedding_dim, 
                                name='embedding_layer')(input_context)
    
    # Capa Lambda para calcular el promedio de los embeddings del contexto
    average_embedding = Lambda(lambda x: K.mean(x, axis=1), name='average')(embedding_layer)
    
    # Capa Dense con activación softmax para predecir la palabra central
    output = Dense(vocab_size, activation='softmax', name='output')(average_embedding)
    
    # Definir el modelo
    model = Model(inputs=input_context, outputs=output)
    
    # Compilar el modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
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



def preparar_datos(token_ids, vocab_size, window_size, test_size=0.2, dev_size=0.1):
    muestras = generar_muestras(token_ids, window_size)
    train_muestras, test_muestras = train_test_split(muestras, test_size=test_size + dev_size, random_state=42)
    train_muestras, dev_muestras = train_test_split(train_muestras, test_size=dev_size / (test_size + dev_size), random_state=42)
    
    def convertir_muestras(muestras):
        X = np.array([contexto for contexto, _ in muestras])
        y = np.array([objetivo for _, objetivo in muestras])
        y_one_hot = tf.keras.utils.to_categorical(y, num_classes=vocab_size)
        return X, y_one_hot
    
    return convertir_muestras(train_muestras), convertir_muestras(dev_muestras), convertir_muestras(test_muestras)

def entrenar_modelo(token_ids, vocab_size, window_size=5, embedding_dim=100, epochs=10, batch_size=64):
    (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = preparar_datos(token_ids, vocab_size, window_size)
    model = build_model(vocab_size, embedding_dim, window_size - 1)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_dev, y_dev))
    return model, X_test, y_test, model.get_layer("embedding_layer").get_weights()[0]

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



"""
# Cargamos el texto de Harry Potter
ruta_archivo = os.path.join("datasets", "harry_potter_and_the_philosophers_stone.txt")
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
print(f"  - Longitud de la secuencia de IDs: {len(token_ids_hp)}")"""


###PROBLEMA NO PONE LOS INDICES EN ORDEN EL:2 LA:1

# Datos de prueba: Secuencia de ejemplo (un texto corto para prueba)
texto_prueba = "el gato se sentó en la alfombra mientras la luna brillaba"

# Tokenizar el texto
tokenizer = Tokenizer()
tokenizer.fit_on_texts([texto_prueba])
token_ids = tokenizer.texts_to_sequences([texto_prueba])[0]
word_index = tokenizer.word_index
index_word = {v: k for k, v in word_index.items()}
vocab_size = len(word_index) + 1  # Para incluir el índice 0

print(f"Vocabulario: {word_index}")
print(f"Secuencia de tokens: {token_ids}")
print(f"Índice de palabras: {index_word}")

# Parámetros
window_size = 3  # Definir una ventana de contexto de tamaño 3

# Generar muestras de contexto y palabra objetivo
muestras = generar_muestras(token_ids, window_size)
print(f"Muestras generadas: {muestras}")

# Preparar los datos para el modelo sin entrenarlo
# Aquí simplemente usamos las muestras generadas y mostramos el resultado
X, y = zip(*muestras)  # Separar las muestras en X (contextos) y y (palabras objetivo)
print(f"Contextos de entrenamiento (X): {X}")
print(f"Palabras objetivo (y): {y}")

# Simular el entrenamiento del modelo con la capa de embeddings
# Crear un modelo vacío solo para obtener la capa de embeddings
embedding_dim = 10  # Dimensión de los embeddings
model = build_model(vocab_size=vocab_size, embedding_dim=embedding_dim, context_size=window_size - 1)

# Como no entrenamos el modelo, inicializamos los pesos aleatoriamente
embedding_layer = model.get_layer("embedding_layer").get_weights()[0]

# Calcular similitudes semánticas para las palabras objetivo
palabras_objetivo = ['gato', 'luna']  # Usamos algunas palabras del texto de prueba
resultados_similitud = calcular_similitud(embedding_layer, word_index, index_word, palabras_objetivo)

print(f"Similitudes semánticas para las palabras objetivo: {resultados_similitud}")

# Visualizar los embeddings de las palabras objetivo usando t-SNE
visualizar_tsne(embedding_layer, word_index, palabras_objetivo, titulo="Embeddings de palabras objetivo")






"""
print("  - Ejemplo del vocabulario:")
verified_examples = [f"'{word}': {index}" for word, index in word_index_hp.items() if "'" in word or any(c in 'áéíóúüñ' for c in word)][:15]
print("    " + ", ".join(verified_examples))

palabras_hp_objetivo = ["harry", "potter", "ron", "hermione", "dumbledore", "magic", "stone", "wizard", "hogwarts", "hagrid", "sorcerer's", "didn't"]
ventana_hp = 5

contextos_hp = obtener_contextos_pretokenizados(
    token_ids=token_ids_hp,
    palabras_objetivo=palabras_hp_objetivo,
    window_size=ventana_hp,
    word_index=word_index_hp,
    index_word=index_word_hp
)

print("\n--- Contextos Encontrados en Harry Potter ---")

if not contextos_hp:
    print("No se encontraron contextos para ninguna de las palabras objetivo especificadas.")
else:
    for palabra, lista_contextos in sorted(contextos_hp.items()):
        print(f"  '{palabra}':")
        for i, contexto in enumerate(lista_contextos[:5]):
            print(f"    - Contexto {i+1}: {contexto}")
        if len(lista_contextos) > 5:
            print(f"    ... (y {len(lista_contextos) - 5} contextos más)")

print("Ejemplo de palabras en el índice:")
for word in ["sorcerer's", "didn't"]:
    print(f"'{word}': {word_index_hp.get(word, 'No encontrado')}")
"""