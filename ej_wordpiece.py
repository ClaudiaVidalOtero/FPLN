# CLAUDIA

import collections
import re
from ej_tresprimeros import tokenize_by_spaces

def train_wordpiece(text, vocab_size=100):

    # Pre-tokenización por espacios
    words = tokenize_by_spaces(text)
    print(words)

    # Inicialización del vocabulario
    vocab = {"[UNK]": 0}
    subwords = collections.defaultdict(int)
    segmented_words = {}
    
    # Segmentación inicial en caracteres
    for word in words:
        chars = list(word)
        segmented_words[word] = [chars[0]] + ["##" + c for c in chars[1:]]

        for subword in segmented_words[word]:
            subwords[subword] += 1
    print(subwords)
    
    # Iterativamente fusionar pares más frecuentes
    while len(vocab) < vocab_size:
        pairs = collections.defaultdict(int)
        
        # Contar pares de subpalabras consecutivas
        for segments in segmented_words.values():
            for i in range(len(segments) - 1):
                pairs[(segments[i], segments[i + 1])] += 1
        
        if not pairs:
            break
        
        # Seleccionar el par más frecuente
        best_pair = max(pairs, key=pairs.get)
        new_token = best_pair[0] + best_pair[1][2:] if best_pair[1].startswith("##") else best_pair[0] + best_pair[1]
        
        # Actualizar vocabulario
        vocab[new_token] = pairs[best_pair]
        
        # Actualizar segmentación de palabras
        for word in segmented_words:
            segments = segmented_words[word]
            new_segments = []
            i = 0
            while i < len(segments):
                if i < len(segments) - 1 and (segments[i], segments[i + 1]) == best_pair:
                    new_segments.append(new_token)
                    i += 2
                else:
                    new_segments.append(segments[i])
                    i += 1
            segmented_words[word] = new_segments
    
    return vocab, segmented_words

def tokenize_wordpiece(text, vocab):
    words = text.split()
    tokens = []
    
    for word in words:
        subword_tokens = []
        while word:
            for i in range(len(word), 0, -1):
                sub = word[:i] if word in vocab else "##" + word[:i]
                if sub in vocab:
                    subword_tokens.append(sub)
                    word = word[i:]
                    break
            else:
                subword_tokens.append("[UNK]")
                break
        
        tokens.extend(subword_tokens)
    
    return tokens

# Ejemplo de uso
corpus = "El gato duerme tranquilo. El perro ladra al cartero."
vocab, segmented_words = train_wordpiece(corpus, vocab_size=50)

text = "El gato corre rápido."
tokens = tokenize_wordpiece(text, vocab)
print("Tokens:", tokens)
