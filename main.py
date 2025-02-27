import argparse
import matplotlib.pyplot as plt
from ej_tresprimeros import tokenize_by_ngrams, tokenize_by_punctuation, tokenize_by_spaces
from ej_wordpiece import train_wordpiece, tokenize_wordpiece

# usar con python main.py --print_wordpiece   o   python main.py --compare_methods   


# Función para entrenar WordPiece con distintos tamaños de vocabulario
def print_wordpiece():
    
    train_path = "materiales-20250207/training_sentences.txt"
    test_path = "materiales-20250207/test_sentences.txt"

    # Leer corpus de entrenamiento
    with open(train_path, "r", encoding="utf-8") as file:
        corpus_train = " ".join(line.strip() for line in file.readlines())

    # Leer corpus de prueba
    with open(test_path, "r", encoding="utf-8") as file:
        corpus_test = " ".join(line.strip() for line in file.readlines())



    for vocab_size in [100, 150, 200]:
        print("_____________________________________________________________________")
        print(f"\n Entrenando WordPiece con vocabulario de tamaño {vocab_size}...")
        print("_____________________________________________________________________")

        vocab = train_wordpiece(corpus_train, max_vocab_size=vocab_size)

        print(f"Vocabulario final ({vocab_size} palabras):", vocab)
        print("_____________________________________________________________________")

        print("\nTokenización en conjunto de entrenamiento:")
        print(tokenize_wordpiece(corpus_train, vocab))
        print("_____________________________________________________________________")

        print("\nTokenización en conjunto de prueba:")
        print(tokenize_wordpiece(corpus_test, vocab))


# Función para comparar la evolución del vocabulario entre métodos de tokenización
def compare_tokenization_methods():
    majesty_path = "materiales-20250207/majesty_speeches.txt"

    # Leer corpus de majesty speeches
    with open(majesty_path, "r", encoding="utf-8") as file:
        corpus_majesty = [line.strip() for line in file.readlines() if line.strip()]

    tokenization_methods = {
        "Espacios": tokenize_by_spaces,
        "Puntuación": tokenize_by_punctuation,
        "N-gramas (n=2)": lambda text: tokenize_by_ngrams(text, 2),
        "WordPiece": lambda text: train_wordpiece(text, max_vocab_size=300),
    }

    vocab_sizes = {method: [] for method in tokenization_methods}
    sentence_counts = list(range(100, len(corpus_majesty), 100))

    for count in sentence_counts:
        partial_corpus = " ".join(corpus_majesty[:count])

        for method, tokenizer in tokenization_methods.items():
            if method == "WordPiece":
                vocab = tokenizer(partial_corpus)
                vocab_sizes[method].append(len(vocab))
            else:
                tokens = tokenizer(partial_corpus)
                vocab_sizes[method].append(len(set(tokens)))

    # Graficar evolución del vocabulario
    plt.figure(figsize=(10, 5))
    for method, sizes in vocab_sizes.items():
        plt.plot(sentence_counts, sizes, marker="o", linestyle="-", label=method)

    plt.xlabel("Número de oraciones procesadas")
    plt.ylabel("Tamaño del vocabulario")
    plt.title("Comparación de evolución del vocabulario entre métodos de tokenización")
    plt.legend()
    plt.grid()
    plt.show()


# Configuración de la línea de comandos
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutar experimentos de tokenización.")
    parser.add_argument(
        "--print_wordpiece",
        action="store_true",
        help="Entrenar WordPiece con diferentes tamaños de vocabulario",
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
    else:
        print("Debes especificar una opción: --print_wordpiece o --compare_methods")
