import math
from collections import defaultdict, Counter


class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = {}
        self.word_counts = {}
        self.class_word_totals = {}
        self.vocab = set()

    def preprocess(self, text):
        return text.lower().split()

    def train(self, documents, labels):
        class_doc_counts = Counter(labels)
        total_docs = len(labels)

        # Compute Priors P(c)
        for c in class_doc_counts:
            self.class_priors[c] = class_doc_counts[c] / total_docs

        # Count words per class
        self.word_counts = {c: defaultdict(int) for c in class_doc_counts}
        self.class_word_totals = {c: 0 for c in class_doc_counts}

        for text, c in zip(documents, labels):
            words = self.preprocess(text)
            for w in words:
                self.word_counts[c][w] += 1
                self.class_word_totals[c] += 1
                self.vocab.add(w)

    def predict(self, document):
        words = self.preprocess(document)
        V = len(self.vocab)

        scores = {}
        for c in self.class_priors:
            log_prob = math.log(self.class_priors[c])

            for w in words:
                # Laplace smoothing
                count = self.word_counts[c][w] + 1
                total = self.class_word_totals[c] + V
                log_prob += math.log(count / total)

            scores[c] = log_prob

        return max(scores, key=scores.get)
    
    
    
    
#  Task 2
def evaluate(model, test_docs, test_labels):
    pred = [model.predict(d) for d in test_docs]

    tp = sum(p == "pos" and t == "pos" for p, t in zip(pred, test_labels))
    fp = sum(p == "pos" and t == "neg" for p, t in zip(pred, test_labels))
    fn = sum(p == "neg" and t == "pos" for p, t in zip(pred, test_labels))

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if precision+recall else 0

    return precision, recall, f1


# Example usage
train_docs = ["good movie", "excellent film", "bad movie", "terrible film"]
train_labels = ["pos", "pos", "neg", "neg"]

test_docs  = ["good film", "terrible movie"]
test_labels = ["pos", "neg"]

model = NaiveBayesClassifier()
model.train(train_docs, train_labels)

P, R, F = evaluate(model, test_docs, test_labels)

print("Precision:", P)
print("Recall:", R)
print("F1 Score:", F)

