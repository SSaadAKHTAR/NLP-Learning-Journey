from task1 import min_edit_distance

def load_vocab(path="Lab6/vocab.txt"):
    with open(path, "r") as f:
        return [w.strip() for w in f.readlines()]

def get_suggestions(prefix, vocab):
    results = []
    for v in vocab:
        if v.startswith(prefix[0]):  # small optimization
            d = min_edit_distance(prefix, v)
            results.append((v, d))

    results.sort(key=lambda x: x[1])
    return results[:10]

if __name__ == "__main__":
    vocab = load_vocab()
    typed = ""

    print("Start typing. Press Enter to quit.")

    while True:
        ch = input("Next char: ").strip()
        if ch == "":
            break

        typed += ch
        print(f"\nTyped: {typed}")
        print("Suggestions:")
        for w, d in get_suggestions(typed, vocab):
            print(f"  {w} ")
