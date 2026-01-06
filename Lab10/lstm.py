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

def tanh(x):
    return math.tanh(x)


hidden_size = 6
input_size = len(text_to_vector("test"))

def init_gate():
    return [[random.uniform(-0.1, 0.1) for _ in range(input_size + hidden_size)]
            for _ in range(hidden_size)]

Wf = init_gate()
Wi = init_gate()
Wo = init_gate()
Wc = init_gate()

bf = [0.0] * hidden_size
bi = [0.0] * hidden_size
bo = [0.0] * hidden_size
bc = [0.0] * hidden_size

Wy = [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
by = 0.0

for epoch in range(30):
    total_loss = 0

    for text, label in dataset:
        h = [0.0] * hidden_size
        c = [0.0] * hidden_size

        for word in text.split():
            x = text_to_vector(word)
            combined = x.tolist() + h

            f = [sigmoid(dot(Wf[i], combined) + bf[i]) for i in range(hidden_size)]
            i_gate = [sigmoid(dot(Wi[i], combined) + bi[i]) for i in range(hidden_size)]
            o = [sigmoid(dot(Wo[i], combined) + bo[i]) for i in range(hidden_size)]
            c_hat = [tanh(dot(Wc[i], combined) + bc[i]) for i in range(hidden_size)]

            c = [f[i] * c[i] + i_gate[i] * c_hat[i] for i in range(hidden_size)]
            h = [o[i] * tanh(c[i]) for i in range(hidden_size)]

        output = sigmoid(dot(Wy, h) + by)
        
        error = output - label

        for i in range(hidden_size):
            Wy[i] -= 0.01 * error * h[i]

        by -= 0.01 * error


        loss = -(label * math.log(output + 1e-9) +
                 (1 - label) * math.log(1 - output + 1e-9))
        total_loss += loss

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")
