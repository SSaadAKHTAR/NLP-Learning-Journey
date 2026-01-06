import math
import os
import re
from collections import defaultdict


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    return tokens



def load_documents(folder):
    documents = {}
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        # Skip directories if any
        if not os.path.isfile(file_path):
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            documents[filename] = preprocess(f.read())

    return documents



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH = os.path.join(BASE_DIR, "docs")

docs = load_documents(DOCS_PATH)
N = len(docs)

if N == 0:
    raise ValueError("No documents found in docs folder!")



vocab = set()
inverted_index = defaultdict(dict)

for doc_id, words in docs.items():
    for word in set(words):
        vocab.add(word)
        inverted_index[word][doc_id] = words.count(word)

vocab = list(vocab)



idf = {}
for term in vocab:
    df = len(inverted_index[term])
    idf[term] = math.log10(N / df)



def tf_idf_vector(words):
    vector = {}
    for term in vocab:
        tf = words.count(term)
        if tf > 0:
            tf = math.log10(tf + 1)
        vector[term] = tf * idf.get(term, 0)
    return vector


doc_vectors = {
    doc_id: tf_idf_vector(words)
    for doc_id, words in docs.items()
}



def cosine_similarity(vec1, vec2):
    dot_product = sum(vec1[t] * vec2[t] for t in vocab)
    mag1 = math.sqrt(sum(vec1[t] ** 2 for t in vocab))
    mag2 = math.sqrt(sum(vec2[t] ** 2 for t in vocab))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)



def search(query):
    query_tokens = preprocess(query)
    query_vector = tf_idf_vector(query_tokens)

    scores = {}
    for doc_id, doc_vector in doc_vectors.items():
        scores[doc_id] = cosine_similarity(query_vector, doc_vector)

    ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_results



query = "sports is good for health"
results = search(query)

print("Query:", query)
print("\nTop Matching Documents:")
for doc, score in results:
    print(doc, "->", round(score, 4))
