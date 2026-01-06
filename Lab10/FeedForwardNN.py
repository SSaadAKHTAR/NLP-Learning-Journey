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



# Network sizes
input_size = len(text_to_vector("test"))
hidden_size = 10
output_size = 1

# Initialize weights
W1 = [[random.uniform(-0.1, 0.1) for _ in range(input_size)] for _ in range(hidden_size)]
b1 = [0.0] * hidden_size

W2 = [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]
b2 = 0.0

learning_rate = 0.1
epochs = 3000

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

for epoch in range(epochs):
    total_loss = 0

    for text, label in dataset:
        x = text_to_vector(text)

        hidden = [sigmoid(dot(W1[i], x) + b1[i]) for i in range(hidden_size)]
        output = sigmoid(dot(W2, hidden) + b2)

        loss = -(label * math.log(output + 1e-9) + (1 - label) * math.log(1 - output + 1e-9))
        total_loss += loss

        error = output - label

        for i in range(hidden_size):
            W2[i] -= learning_rate * error * hidden[i]
        b2 -= learning_rate * error

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")