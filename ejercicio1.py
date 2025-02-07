from nltk.util import ngrams
import os
import re
ruta = "materiales-20250207/training_sentences.txt"

# Abrir el archivo y leer las líneas
with open(ruta, "r", encoding="utf-8") as file:
    lines = file.readlines()  

text = " ".join(line.strip() for line in lines)



def tokenize_by_spaces(text):
    return text.split()

def tokenize_by_punctuation(text):
    punctuation = r"\w+|[.,!?;:()\[\]{}\"'`~<>@#$%^&*-+=/\\]"
    return re.findall(punctuation, text)

def tokenize_by_ngrams(text, n):

    words = tokenize_by_spaces(text)
    return list(ngrams(words, n))


print("Tokenización por espacios:", tokenize_by_spaces(text))
print("Tokenización por signos de puntuación:", tokenize_by_punctuation(text))
print("Tokenización por bigramas:", tokenize_by_ngrams(text, 2))