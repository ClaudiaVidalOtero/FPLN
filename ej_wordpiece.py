import collections
from tqdm import tqdm

def train_wordpiece(corpus, max_vocab_size=100, num_epochs=30, min_pair_freq=1):
    vocab = {"[UNK]": 0}
    word_freqs = collections.Counter(corpus.split())
    subwords = {}
    updated_words = {}  

    # Inicializamos vocabulario con caracteres individuales
    for word, freq in word_freqs.items():
        tokens = [word[0]] + ["##" + c for c in word[1:]]
        updated_words[word] = tokens
        for token in tokens:
            subwords[token] = subwords.get(token, 0) + freq

    vocab.update(subwords)  

    progress_bar = tqdm(total=num_epochs, desc="Training WordPiece", unit="epoch")

    for epoch in range(num_epochs):
        pairs = collections.defaultdict(int)

        # Contar pares de subpalabras
        for word, symbols in updated_words.items():
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += word_freqs[word]  

        if not pairs:
            break

        # Elegir el par más frecuente por encima del umbral 
        best_pair = max(pairs, key=pairs.get)
        if pairs[best_pair] < min_pair_freq:
            print("Detenido temprano: No hay más pares frecuentes.")
            break

        new_token = best_pair[0] + best_pair[1].replace("##", "")

        if new_token in vocab:
            continue  

        vocab[new_token] = pairs[best_pair]

        # Reemplazar el par con el nuevo token
        for word, symbols in updated_words.items():
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                    new_symbols.append(new_token)
                    i += 2  
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            updated_words[word] = new_symbols

        print(f"Iteración {epoch+1}: Fusionando {best_pair} -> {new_token}")

        # Si alcanzamos el tamaño máximo del vocabulario, detenemos el entrenamiento
        if len(vocab) >= max_vocab_size:
            print("Límite de vocabulario alcanzado.")
            break

        progress_bar.update(1)

    progress_bar.close()
    return vocab

def tokenize_wordpiece(sentence, vocab):
    tokens = []
    words = sentence.split()

    for word in words:
        sub_tokens = []
        start = 0

        while start < len(word):
            match_found = False

            for end in range(len(word), start, -1):
                subword = word[start:end]
                subword_with_prefix = "##" + subword if start > 0 else subword  # Prefijo "##" para subpalabras

                if subword_with_prefix in vocab:
                    sub_tokens.append(subword_with_prefix)
                    start = end
                    match_found = True
                    break
                elif subword in vocab and start == 0:
                    sub_tokens.append(subword)
                    start = end
                    match_found = True
                    break

            if not match_found:
                sub_tokens.append("[UNK]")
                start += 1  

        tokens.extend(sub_tokens)

    return tokens


# Cargamos corpus de entrenamiento desde archivo
ruta = "materiales-20250207/training_sentences.txt"

with open(ruta, "r", encoding="utf-8") as file:
    lines = file.readlines()

corpus_train = " ".join(line.strip() for line in lines)

vocab = train_wordpiece(corpus_train, max_vocab_size=150, num_epochs=100)

print("\n📌 Vocabulario final:", vocab)


# Cargamos corpus de test desde archivo
ruta2 = "materiales-20250207/test_sentences.txt"

with open(ruta2, "r", encoding="utf-8") as file:
    lines = file.readlines()

corpus_test = " ".join(line.strip() for line in lines)

tokenized_sentence = tokenize_wordpiece(corpus_test, vocab)

print("\n📝 Tokenización:", tokenized_sentence)

