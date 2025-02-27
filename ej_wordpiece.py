import collections
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_wordpiece(corpus, max_vocab_size=100, min_pair_freq=1):
    vocab = {"[UNK]": 0}
    word_freqs = collections.Counter(corpus.split())
    subwords = {}
    updated_words = {}
    
    # Inicializar vocabulario con caracteres individuales
    for word, freq in word_freqs.items():
        tokens = [word[0]] + ["##" + c for c in word[1:]]
        updated_words[word] = tokens
        for token in tokens:
            subwords[token] = subwords.get(token, 0) + freq
    
    vocab.update(subwords)
    print("Entrenando el algoritmo WordPiece...")
    while len(vocab) < max_vocab_size:
        
        pairs = collections.defaultdict(int)
        
        # Contar pares de subpalabras
        for word, symbols in updated_words.items():
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += word_freqs[word]
        
        if not pairs:
            break
        
        # Elegir el par más frecuente
        best_pair = max(pairs, key=pairs.get)
        if pairs[best_pair] < min_pair_freq:
            print("Detenido temprano: No hay más pares frecuentes.")
            break
        
        # Fusionar eliminando ## que no van al principio
        new_token = best_pair[0].replace("##", "") + best_pair[1].replace("##", "")
        if not best_pair[0].startswith("##"):
            new_token = best_pair[0] + best_pair[1].replace("##", "")
        
        if new_token in vocab:
            continue
        
        vocab[new_token] = pairs[best_pair]
        
        # Actualizar segmentaciones
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
        
        if len(vocab) >= max_vocab_size:
            print("Límite de vocabulario alcanzado.")
            break
    
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
                # Si alguna parte de la palabra no está en el vocabulario, se descarta todo y se usa [UNK]
                sub_tokens = ["[UNK]"]
                break  
        
        # Verificar si todos los sub_tokens tienen solo una letra o número
        if all(len(token.replace("##", "")) == 1 for token in sub_tokens) or all(token.replace("##", "").isdigit() for token in sub_tokens):
            tokens.append("[UNK]")
        else:
            tokens.append(" ".join(sub_tokens))

    return tokens

