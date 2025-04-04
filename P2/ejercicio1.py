import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# PARÁMETROS A A ELEGIR POR EL USUARIO:
window_size = 5
embedding_dim = 100
epochs = 30
batch_size = 128


# Construye y compila el modelo CBOW con una capa de embeddings y softmax
def build_model(vocab_size, embedding_dim, window_size):
    input_context = layers.Input(shape=(window_size-1,), name='context_input')  # Entrada: palabras de contexto
    embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='embedding')(input_context)
    average_layer = layers.Lambda(lambda x: K.mean(x, axis=1), name='average_embedding')(embedding_layer)   # Promedio de embeddings del contexto
    output_layer = layers.Dense(units=vocab_size, activation='softmax', name='output_dense_softmax')(average_layer)     # Predicción de palabra objetivo

    model = keras.Model(inputs=input_context, outputs=output_layer, name='Modelo_CBOW_1')
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])   # Configura el optimizador y pérdida

    return model


# Genera muestras de entrenamiento usando ventanas deslizantes
def generar_muestras(token_ids, window_size):
    context_half_size = (window_size - 1) // 2
    return [(token_ids[i - context_half_size:i] + token_ids[i + 1:i + 1 + context_half_size], token_ids[i])
            for i in range(context_half_size, len(token_ids) - context_half_size)]  # Cada muestra: (contexto, objetivo)


# Prepara los datos para el entrenamiento (X contexto, y palabra objetivo)
def preparar_datos(token_ids, window_size):
    muestras = generar_muestras(token_ids, window_size)

    def convertir_muestras(muestras):
        X = np.array([contexto for contexto, _ in muestras])  # Contextos como entrada
        y = np.array([objetivo for _, objetivo in muestras])  # Palabras objetivo como salida
        return X, y

    return convertir_muestras(muestras)


# Entrena el modelo CBOW y guarda los embeddings antes y después
def entrenar_modelo(token_ids, vocab_size, window_size=5, embedding_dim=100, epochs=20, batch_size=128):
    (X, y) = preparar_datos(token_ids, window_size)

    # División en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    model = build_model(vocab_size, embedding_dim, window_size)
    model.summary()  

    embedding_inicial = model.get_layer("embedding").get_weights()[0]  # Extrae los embeddings iniciales
    visualizar_tsne(embedding_inicial, word_index_hp, palabras_objetivo, titulo="Embeddings Iniciales con t-SNE")

    print(f"\nIniciando entrenamiento por hasta {epochs} épocas...")

    # Callback para detener entrenamiento si no mejora la pérdida de validación
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],  
        verbose=1
    )

    embedding_weights = model.get_layer("embedding").get_weights()[0]  # Extraemos los embeddings aprendidos tras el entrenamiento

    print(f"Forma de la matriz de embeddings: {embedding_weights.shape}")

    return model, X_train, y_train, embedding_weights


# Calcula las palabras más similares (por coseno) a las palabras objetivo
def calcular_similitud(embedding_layer, word_index, index_word, palabras_objetivo, top_n=10, mostrar=True):
    resultados = {}
    for palabra in palabras_objetivo:
        if palabra in word_index:
            palabra_id = word_index[palabra]
            vector = embedding_layer[palabra_id].reshape(1, -1)  # Vector de la palabra
            similitudes = cosine_similarity(vector, embedding_layer)[0]  # Similitud con todas las demás
            palabras_mas_similares = np.argsort(similitudes)[::-1][1:top_n + 1]  # Top N similares (excepto ella misma)
            similares = [index_word.get(idx, f"<ID_{idx}?>") for idx in palabras_mas_similares]
            resultados[palabra] = similares
            if mostrar:
                print(f"{palabra}: {', '.join(similares)}")
        else:
            if mostrar:
                print(f"{palabra}: [Palabra no encontrada en el vocabulario]")
    return resultados


# Visualiza los embeddings con t-SNE para palabras objetivo
def visualizar_tsne(embedding_layer, word_index, palabras_objetivo, titulo="Embeddings con t-SNE"):

    ids = [word_index[palabra] for palabra in palabras_objetivo if palabra in word_index]
    if len(ids) < 2:
        print("No hay suficientes palabras para aplicar t-SNE")
        return

    vectores = np.array([embedding_layer[i] for i in ids])  # Extrae vectores de embeddings
    palabras = [palabra for palabra in palabras_objetivo if palabra in word_index]

    # Aplica reducción de dimensionalidad
    tsne = TSNE(n_components=2, perplexity=min(5, len(vectores)-1), random_state=42)
    vectores_2d = tsne.fit_transform(vectores)

    plt.figure(figsize=(10, 6))
    for i, palabra in enumerate(palabras):
        plt.scatter(vectores_2d[i, 0], vectores_2d[i, 1], label=palabra)
        plt.text(vectores_2d[i, 0], vectores_2d[i, 1], palabra, fontsize=12)
    plt.title(titulo)
    plt.show()


# CARGAMOS LOS ARCHIVOS NECESARIOS

# Texto a tokenizar
ruta_archivo = os.path.join("harry_potter_and_the_philosophers_stone.txt")
with open(ruta_archivo, 'r', encoding='utf-8') as file:
    texto_harry_potter = file.read()
print(f"Archivo '{ruta_archivo}' leído.")

# Palabras objetivo
nombre_archivo = os.path.join("target_words_harry_potter.txt")
def leer_palabras_desde_archivo(nombre_archivo):
    with open(nombre_archivo, "r", encoding="utf-8") as archivo:
        palabras = [linea.strip() for linea in archivo if linea.strip()] 
    return palabras

palabras_objetivo = leer_palabras_desde_archivo(nombre_archivo)
print(palabras_objetivo)


# TOKENIZAMOS EL TEXTO
tokenizer_hp = Tokenizer()
tokenizer_hp.fit_on_texts([texto_harry_potter])
sequences = tokenizer_hp.texts_to_sequences([texto_harry_potter])
token_ids_hp = sequences[0] if sequences else []  # Convierte a IDs numéricos
word_index_hp = tokenizer_hp.word_index  # Diccionario palabra -> ID
index_word_hp = tokenizer_hp.index_word  # Diccionario ID -> palabra
vocab_size_hp = len(word_index_hp) + 1  # Tamaño del vocabulario


# ENTRENAMOS EL MODELO
modelo, X_test, y_test, embedding_layer = entrenar_modelo(token_ids_hp, vocab_size_hp, window_size, embedding_dim, epochs, batch_size)


# VISUALIZAMOS LOS RESULTADOS
visualizar_tsne(embedding_layer, word_index_hp, palabras_objetivo)  #Visualizamos embeddings

print("Similitudes de palabras objetivo:\n")
similares_por_palabra = calcular_similitud(embedding_layer, word_index_hp, index_word_hp, palabras_objetivo) # Visualizamos similitudes de palabras
