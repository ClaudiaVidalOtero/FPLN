from collections import defaultdict
import os

def train_bpe(ruta_archivo, vocab_size=100):
    with open(ruta_archivo, "r", encoding="utf-8") as archivo:
        texto = archivo.read()
    
    corpus_tokenizado = [line.split() for line in texto.strip().split("\n") if line]
    frecuencia = defaultdict(int)
    for oracion in corpus_tokenizado:
        for termino in oracion:
            frecuencia[termino] += 1
    
    segmentacion = {palabra: list(palabra) for palabra in frecuencia}
    reglas_fusion = []
    vocab_size_over_time = [len(set(c for palabra in segmentacion.values() for c in palabra))]

    
    while len(segmentacion) < vocab_size:
        pares = defaultdict(int)
        for palabra, segm in segmentacion.items():
            for i in range(len(segm) - 1):
                pares[(segm[i], segm[i + 1])] += frecuencia[palabra]
        
        if not pares:
            break
        
        par_frecuente = max(pares, key=pares.get)
        nueva_unidad = "".join(par_frecuente)
        reglas_fusion.append((par_frecuente, nueva_unidad))
        
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
    
    return reglas_fusion, vocab_size_over_time

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

if __name__ == "__main__":
    archivo_entrada = "materiales-20250207/training_sentences.txt"
    archivo_test = "materiales-20250207/test_sentences.txt"
 
    vocab_size = 150
    reglas = train_bpe(archivo_entrada, vocab_size)
    
    with open(archivo_test, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                tokens = tokenize_bpe(linea, reglas)
                print(f"Input: '{linea}' -> Tokens: {tokens}")
