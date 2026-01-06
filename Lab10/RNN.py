import spacy
import math
import random

nlp = spacy.load("en_core_web_md")

dataset = [
    ("I love this movie", 1),
    ("This film is terrible", 0),
    ("Amazing experience", 1),
    ("Worst movie ever", 0),
    ("I really enjoyed this", 1),
    ("I hate this film", 0)
]

def text_to_vector(text):
    return nlp(text).vector

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

hidden_size = 8
input_size = len(text_to_vector("test"))

Wx = [[random.uniform(-0.1, 0.1) for _ in range(input_size)] for _ in range(hidden_size)]
Wh = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(hidden_size)]
Wy = [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]

bh = [0.0] * hidden_size
by = 0.0

def tanh(x):
    return math.tanh(x)

for epoch in range(40):
    total_loss = 0

    for text, label in dataset:
        words = text.split()
        h = [0.0] * hidden_size

        for word in words:
            x = text_to_vector(word)
            h = [
                tanh(dot(Wx[i], x) + dot(Wh[i], h) + bh[i])
                for i in range(hidden_size)
            ]
        

        output = sigmoid(dot(Wy, h) + by)
        
        error = output - label

        for i in range(hidden_size):
            Wy[i] -= 0.01 * error * h[i]

        by -= 0.01 * error

        loss = -(label * math.log(output + 1e-9) +
                 (1 - label) * math.log(1 - output + 1e-9))
        total_loss += loss

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")
