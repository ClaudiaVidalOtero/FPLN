
import os
import re
from typing import Dict, List


import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Embedding, Input, Lambda 
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer 

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

def generar_muestras_desde_secuencia(token_ids, window_size):

    # Verificamos que la ventana deslizante cumpla las condiciones requeridas
    if not isinstance(window_size, int) or window_size < 3:
        raise ValueError("window_size debe ser un entero mayor o igual a 3.")
    
    if window_size % 2 == 0:
        raise ValueError("window_size debe ser un número impar (e.g., 3, 5, 7...).")

    context_half_size = (window_size - 1) // 2
    n_tokens = len(token_ids)
    muestras = []

    # En caso de que no haya suficiente número de tokens
    if n_tokens < window_size:
        return []

    for i in range(context_half_size, n_tokens - context_half_size):
        objetivo_id = token_ids[i]

        contexto_antes = token_ids[i - context_half_size : i]
        contexto_despues = token_ids[i + 1 : i + context_half_size + 1]

        contexto_ids = contexto_antes + contexto_despues

        # Si el tamaño del contexto es suficiente
        if len(contexto_ids) == 2 * context_half_size:
             muestras.append((contexto_ids, objetivo_id))

    return muestras

def obtener_contextos_pretokenizados(token_ids, palabras_objetivo, window_size, word_index, index_word):
    
    # En caso de que la palabra no esté en el índice
    if not token_ids:
        print("Advertencia: La secuencia de token_ids está vacía.")
        return {palabra.lower().strip(): [] for palabra in palabras_objetivo if palabra.strip()}

    # Generamos muestras del contexto 
    try:
        muestras_enteras = generar_muestras_desde_secuencia(token_ids, window_size)
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
print(f"  - Longitud de la secuencia de IDs: {len(token_ids_hp)}")

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
