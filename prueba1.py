import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Lambda, Dense
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# 1. Cargar y tokenizar el dataset
with open("harry_potter_and_the_philosophers_stone.txt", "r", encoding="utf-8") as f:
    text = f.read()

# El texto ya viene preprocesado (minúsculas, sin tildes, etc.)
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
sequences = tokenizer.texts_to_sequences([text])[0]
vocab_size = len(tokenizer.word_index) + 1  # +1 para incluir el índice 0

print("Tamaño del vocabulario:", vocab_size)
print("Número de tokens:", len(sequences))

# 2. Generar muestras de entrenamiento utilizando una ventana de 5 (2 palabras anteriores y 2 posteriores)
window_size = 2  # 2 palabras antes y 2 palabras después; total ventana = 5 (con la central)
contexts = []
targets = []

# Recorrer el corpus, omitiendo los extremos donde no se puede formar la ventana
for i in range(window_size, len(sequences) - window_size):
    context = []
    # Extraemos el contexto: omitimos la palabra central
    for j in range(i - window_size, i + window_size + 1):
        if j == i:
            continue
        context.append(sequences[j])
    contexts.append(context)
    targets.append(sequences[i])

contexts = np.array(contexts)
targets = np.array(targets)

print("Número de muestras de entrenamiento:", contexts.shape[0])

# 3. Definir la arquitectura del modelo
embedding_dim = 100  

# Capa de entrada: cada muestra es un vector de 4 IDs (contexto)
context_input = Input(shape=(4,), name='context_input')

# Capa de Embedding: mapea cada ID a un vector de dimensión embedding_dim
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='embedding')
context_embeddings = embedding_layer(context_input)

# Capa Lambda para calcular la media de los embeddings (promedio a lo largo del eje 1)
context_mean = Lambda(lambda x: K.mean(x, axis=1), name='average')(context_embeddings)

# Capa densa de salida: predice la palabra central con softmax
output = Dense(vocab_size, activation='softmax', name='output')(context_mean)

# Definir y compilar el modelo
model = Model(inputs=context_input, outputs=output)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 4. Dividir el dataset en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(contexts, targets, test_size=0.1, random_state=42)

# 5. Definir callbacks para EarlyStopping y ReduceLROnPlateau
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1)

# 6. Entrenar el modelo con los callbacks
epochs = 30  # Puedes aumentar el número total de épocas; el EarlyStopping detendrá el entrenamiento si no hay mejora
batch_size = 128

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stop, reduce_lr]
)

# 7. Extracción de los embeddings entrenados
embeddings = model.get_layer('embedding').get_weights()[0]
print("Shape de los embeddings:", embeddings.shape)

# Ejemplo: obtener el embedding de una palabra (por su ID)
word_id = 100
if word_id < vocab_size:
    print("Embedding para la palabra con ID", word_id, ":", embeddings[word_id])
else:
    print("El ID", word_id, "excede el tamaño del vocabulario.")
