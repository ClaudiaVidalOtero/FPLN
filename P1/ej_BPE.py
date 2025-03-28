
def train_bpe(ruta_archivo, vocab_size=100):
    from collections import defaultdict

    with open(ruta_archivo, "r", encoding="utf-8") as archivo:
        texto = archivo.read()

    # Tokenizar corpus en palabras
    corpus_tokenizado = [line.split() for line in texto.strip().split("\n") if line]

    # Contar la frecuencia de cada palabra en el corpus
    frecuencia = defaultdict(int)
    for oracion in corpus_tokenizado:
        for termino in oracion:
            frecuencia[termino] += 1

    # Inicializar cada palabra como una lista de caracteres
    segmentacion = {palabra: list(palabra) for palabra in frecuencia}
    reglas_fusion = []
    vocab_growth = []  

    while len(vocab_growth) < vocab_size:  
        pares = defaultdict(int)

        # Contar frecuencia de pares de caracteres consecutivos
        for palabra, segm in segmentacion.items():
            for i in range(len(segm) - 1):
                pares[(segm[i], segm[i + 1])] += frecuencia[palabra]

        if not pares:
            break  # No hay más pares para fusionar

        # Encontrar el par más frecuente
        par_frecuente = max(pares, key=pares.get)
        nueva_unidad = "".join(par_frecuente)  # Fusionar caracteres

        # Guardar la regla de fusión
        reglas_fusion.append((par_frecuente, nueva_unidad))

        nueva_segmentacion = {}

        # Actualizar segmentación de palabras
        for palabra, segm in segmentacion.items():
            nueva_seg = []
            i = 0
            while i < len(segm):
                if i < len(segm) - 1 and (segm[i], segm[i + 1]) == par_frecuente:
                    nueva_seg.append(nueva_unidad)
                    i += 2  # Saltar el par fusionado
                else:
                    nueva_seg.append(segm[i])
                    i += 1
            nueva_segmentacion[palabra] = nueva_seg

        segmentacion = nueva_segmentacion
        

        vocab_set = set()  # Nuevo conjunto de vocabulario
        for palabra, segm in segmentacion.items():
            vocab_set.update(segm)
            
        vocab_growth.append(len(vocab_set))


    return reglas_fusion, vocab_growth



def tokenize_bpe(linea, reglas_fusion):
    palabras = linea.split()
    segmentacion = {termino: list(termino) for termino in palabras}
    
    for par_frecuente, nueva_unidad in reglas_fusion:
        nueva_segmentacion = {}
        for palabra, segm in segmentacion.items():
            nueva_seg = []
            i = 0
            while i < len(segm):
                if i < len(segm) - 1 and (segm[i], segm[i + 1]) == par_frecuente:
                    nueva_seg.append(nueva_unidad)
                    i += 2
                else:
                    nueva_seg.append(segm[i])
                    i += 1
            nueva_segmentacion[palabra] = nueva_seg
        segmentacion = nueva_segmentacion
    
    tokens = [token for palabra in palabras for token in segmentacion[palabra]]
    return tokens
