from task1 import min_edit_distance

def load_vocab(path="Lab6/vocab.txt"):
    with open(path, "r") as f:
        return [w.strip() for w in f.readlines()]

def suggest(word, vocab):
    distances = []
    for v in vocab:
        d = min_edit_distance(word, v)
        distances.append((v, d))

    distances.sort(key=lambda x: x[1])
    return distances[:10]

if __name__ == "__main__":
    vocab = load_vocab()

    word = input("Enter a word: ").strip().lower()

    if word in vocab:
        print("Word is correct!")
    else:
        print("\nTop 10 suggestions:")
        for w, d in suggest(word, vocab):
            print(f"{w} ")