import argparse
import matplotlib.pyplot as plt
from ej_tresprimeros import tokenize_by_ngrams, tokenize_by_punctuation, tokenize_by_spaces
from ej_wordpiece import train_wordpiece, tokenize_wordpiece
from ej_BPE import train_bpe, tokenize_bpe


train_path = "materiales-20250207/training_sentences.txt"
test_path = "materiales-20250207/test_sentences.txt"
majesty_path = "materiales-20250207/majesty_speeches.txt"


# Leer corpus de entrenamiento
with open(train_path, "r", encoding="utf-8") as file:
    corpus_train = [line.strip() for line in file.readlines()]

# Leer corpus de prueba
with open(test_path, "r", encoding="utf-8") as file:
    corpus_test = [line.strip() for line in file.readlines()]

# Leer corpus de majesty speeches
with open(majesty_path, "r", encoding="utf-8") as file:
        corpus_majesty = [line.strip() for line in file.readlines() if line.strip()]



# Función para ejecutar los algoritmos de tokenización por espacios, puntuación y ngramas
def print_test():
    vocab_spaces = tokenize_by_spaces(corpus_test)
    print(f"Vocabulario final tokenizado por espacios:", vocab_spaces)
    print("_____________________________________________________________________")

    vocab_punctuation = tokenize_by_punctuation(corpus_test)
    print(f"Vocabulario final tokenizado por puntuación:", vocab_punctuation)
    print("_____________________________________________________________________")

    vocab_ngrams = tokenize_by_ngrams(corpus_test, 2)
    print(f"Vocabulario final tokenizado por ngramas (n = 2):", vocab_ngrams)


# Función para entrenar el algoritmo WordPiece con distintos tamaños de vocabulario
def print_wordpiece():

      
    for vocab_size in [100, 150, 200]:

        vocab_wordpiece = train_wordpiece(" ".join(corpus_train), max_vocab_size=vocab_size)

        print(f"Vocabulario WordPiece ({vocab_size} palabras):", vocab_wordpiece)
        print("_____________________________________________________________________")


        print("\nTokenización en conjunto de entrenamiento de WordPiece:")
        with open(train_path, 'r', encoding='utf-8') as train_file:
            for line in train_file:
                resultado = tokenize_wordpiece(line, vocab_wordpiece)  
                print(f"Input: '{line}' -> Tokens: {resultado}")
 
        print("_____________________________________________________________________")


        print("\nTokenización en conjunto de prueba de WordPiece:")
         # Abrir el archivo de prueba y leer línea por línea
        with open(test_path, 'r', encoding='utf-8') as test_file:
            for line in test_file:
                resultado = tokenize_wordpiece(line, vocab_wordpiece)  
                print(f"Input: '{line}' -> Tokens: {resultado}")

        print("_____________________________________________________________________")


# Función para entrenar el algoritmo BPE con distintos tamaños de vocabulario
def print_bpe():
        
    for vocab_size in [100, 150, 200]:
        
        vocab_bpe = train_bpe(train_path, vocab_size)

        print(f"Vocabulario final BPE ({vocab_size} palabras):", vocab_bpe)
        print("_____________________________________________________________________")

        print("\nTokenización en conjunto de entrenamiento de BPE:")
        with open(train_path, 'r', encoding='utf-8') as train_file:
            for line in train_file:
                resultado = tokenize_bpe(line, vocab_bpe)  
                print(f"Input: '{line}' -> Tokens: {resultado}")
        print("_____________________________________________________________________")

        print("\nTokenización en conjunto de prueba de BPE:")
        with open(test_path, 'r', encoding='utf-8') as test_file:
            for line in test_file:
                resultado = tokenize_bpe(line, vocab_bpe)  
                print(f"Input: '{line}' -> Tokens: {resultado}")

        print("_____________________________________________________________________")


# Función para comparar la evolución del vocabulario entre métodos de tokenización
def compare_tokenization_methods():
    max_sentences = None
    
    # Convertir corpus_majesty a un solo string para train_wordpiece
    corpus_majesty_text = " ".join(corpus_majesty)
    wordpiece_vocab, wordpiece_vocab_growth = train_wordpiece(corpus_majesty_text, max_vocab_size=3000)
    bpe_rules, bpe_vocab_growth = train_bpe(majesty_path, vocab_size=3000)

    # Diccionario de métodos de tokenización
    tokenizer_methods = {
        "Espacios": tokenize_by_spaces,
        "Puntuación": tokenize_by_punctuation,
        "N-Gramas (2)": lambda text: tokenize_by_ngrams(text, 2),
        "WordPiece": lambda text: tokenize_wordpiece(text, wordpiece_vocab),
        "BPE": lambda text: tokenize_bpe(text, bpe_rules)
    }

    sentences = corpus_majesty
    if max_sentences:
        sentences = sentences[:max_sentences]

    vocab_growth = {method: [] for method in tokenizer_methods.keys()}
    vocab_sets = {method: set() for method in tokenizer_methods.keys()}

    # Calcular evolución del vocabulario
    for i, sentence in enumerate(sentences, start=1):
        for method, tokenizer in tokenizer_methods.items():
            tokens = tokenizer(sentence)
            vocab_sets[method].update(tokens)
            vocab_growth[method].append(len(vocab_sets[method]))

    # Agregar las evoluciones de WordPiece y BPE preentrenadas
    vocab_growth["WordPiece"] = wordpiece_vocab_growth[:len(sentences)]
    vocab_growth["BPE"] = bpe_vocab_growth[:len(sentences)]

    # Graficar todas las evoluciones
    plt.figure(figsize=(10, 6))
    for method, growth in vocab_growth.items():
        plt.plot(range(1, len(growth) + 1), growth, label=method)

    plt.xlabel("Número de oraciones procesadas")
    plt.ylabel("Tamaño del vocabulario (tokens únicos)")
    plt.title("Evolución del tamaño del vocabulario entre métodos de tokenización")
    plt.legend()
    plt.grid()
    plt.show()



# Configuración de la línea de comandos
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutar experimentos de tokenización.")

    parser.add_argument(
        "--print_test",
        action="store_true",
        help="Entrenar los tres primeros algoritmos",
    )
    parser.add_argument(
        "--print_wordpiece",
        action="store_true",
        help="Entrenar WordPiece con diferentes tamaños de vocabulario",
    )
    parser.add_argument(
        "--print_bpe",
        action="store_true",
        help="Comparar la evolución del vocabulario en distintos métodos de tokenización",
    )
    
    parser.add_argument(
        "--compare_methods",
        action="store_true",
        help="Comparar la evolución del vocabulario en distintos métodos de tokenización",
    )
    
    args = parser.parse_args()

    if args.print_wordpiece:
        print_wordpiece()
    elif args.compare_methods:
        compare_tokenization_methods()
    elif args.print_test:
        print_test()
    elif args.print_bpe:
        print_bpe()
    else:
        print("Debes especificar una opción: --print_test, --print_wordpiece, --print_bpe o --compare_methods ")
