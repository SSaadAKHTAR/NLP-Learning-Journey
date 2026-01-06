import spacy
import math
import random


nlp = spacy.load("en_core_web_md")


dataset = [
    ("I love this movie, it is amazing", 1),
    ("This film was terrible and boring", 0),
    ("What a fantastic experience", 1),
    ("Worst movie I have ever seen", 0),
    ("Absolutely wonderful story", 1),
    ("I hate this so much", 0)
]


def text_to_vector(sentence):
    return nlp(sentence).vector



X = [text_to_vector(text) for text, label in dataset]
y = [label for text, label in dataset]


feature_length = len(X[0])     # Size of embedding vector
weights = [0.0] * feature_length
bias = 0.0

learning_rate = 0.1
epochs = 50


def sigmoid(value):
    return 1 / (1 + math.exp(-value))



def dot_product(vec1, vec2):
    return sum(w * x for w, x in zip(vec1, vec2))



for epoch in range(epochs):
    total_loss = 0

    combined_data = list(zip(X, y))
    random.shuffle(combined_data)

    for features, label in combined_data:

        z = dot_product(weights, features) + bias

        prediction = sigmoid(z)

        loss = -(label * math.log(prediction + 1e-9) + (1 - label) * math.log(1 - prediction + 1e-9))
        total_loss += loss

        error = prediction - label

        for i in range(feature_length):
            weights[i] -= learning_rate * error * features[i]

        bias -= learning_rate * error

    print(f"Epoch {epoch + 1}  Loss: {total_loss:.4f}")



def predict_sentiment(sentence):
    vector = text_to_vector(sentence)
    z = dot_product(weights, vector) + bias
    probability = sigmoid(z)

    if probability >= 0.5:
        return "Positive", probability
    else:
        return "Negative", probability



test_sentence = "This movie is awesome"
sentiment, confidence = predict_sentiment(test_sentence)

print("\nTest Sentence:", test_sentence)
print("Predicted Sentiment:", sentiment)
print("Confidence Score:", confidence)
