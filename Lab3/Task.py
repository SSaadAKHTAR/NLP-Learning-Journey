#  Task 1
from collections import Counter
import matplotlib.pyplot as plt
import re
import requests

url = "https://shorturl.at/WeE7u"
text = requests.get(url).text

words_raw = re.findall(r'\b\w+\b', text.lower())
word_freq_raw = Counter(words_raw)

stop_words = {"the", "and", "a", "to", "in", "of", "it", "is", "that", "for", "on", "was", "he", "she", "as", "at"}
filtered_words = [w for w in words_raw if w not in stop_words and len(w) > 2]
word_freq_clean = Counter(filtered_words)

# Plot
plt.figure(figsize=(10, 5))
plt.bar(*zip(*word_freq_raw.most_common(20)))
plt.title("Top 20 Words (Before Preprocessing)")
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(*zip(*word_freq_clean.most_common(20)))
plt.title("Top 20 Words (After Preprocessing)")
plt.show()

#  Task 2
from collections import Counter
from bs4 import BeautifulSoup
import requests
import re
import matplotlib.pyplot as plt

url = "https://blog.python.org/"  
html = requests.get(url).text
text = BeautifulSoup(html, "html.parser").get_text()

words_raw = re.findall(r'\b\w+\b', text.lower())
word_freq_raw = Counter(words_raw)

stop_words = {"the", "and", "a", "to", "in", "of", "it", "is", "that", "for", "on", "was", "he", "she", "as", "at"}
filtered_words = [w for w in words_raw if w not in stop_words and len(w) > 2]
word_freq_clean = Counter(filtered_words)

plt.figure(figsize=(10, 5))
plt.bar(*zip(*word_freq_raw.most_common(20)))
plt.title("Top 20 Words (Before Preprocessing)")
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(*zip(*word_freq_clean.most_common(20)))
plt.title("Top 20 Words (After Preprocessing)")
plt.show()

# Task 3: Word cloud before and after preprocessing
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import requests

url = "https://shorturl.at/WeE7u"
text = requests.get(url).text

words_raw = re.findall(r'\b\w+\b', text.lower())
word_freq_raw = Counter(words_raw)

stop_words = {"the", "and", "a", "to", "in", "of", "it", "is", "that", "for", "on", "was", "he", "she", "as", "at"}
filtered_words = [w for w in words_raw if w not in stop_words and len(w) > 2]
word_freq_clean = Counter(filtered_words)

wordcloud1 = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq_raw)
plt.imshow(wordcloud1, interpolation='bilinear')
plt.title("Word Cloud (Before Preprocessing)")
plt.axis("off")
plt.show()

wordcloud2 = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq_clean)
plt.imshow(wordcloud2, interpolation='bilinear')
plt.title("Word Cloud (After Preprocessing)")
plt.axis("off")
plt.show()

# Task 4
# Task 4: Word cloud for blog HTML
from collections import Counter
from wordcloud import WordCloud
from bs4 import BeautifulSoup
import requests
import re
import matplotlib.pyplot as plt

url = "https://blog.python.org/"  
html = requests.get(url).text
text = BeautifulSoup(html, "html.parser").get_text()

words_raw = re.findall(r'\b\w+\b', text.lower())
word_freq_raw = Counter(words_raw)

stop_words = {"the", "and", "a", "to", "in", "of", "it", "is", "that", "for", "on", "was", "he", "she", "as", "at"}
filtered_words = [w for w in words_raw if w not in stop_words and len(w) > 2]
word_freq_clean = Counter(filtered_words)

wordcloud1 = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq_raw)
plt.imshow(wordcloud1, interpolation='bilinear')
plt.title("Word Cloud (Before Preprocessing)")
plt.axis("off")
plt.show()

wordcloud2 = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq_clean)
plt.imshow(wordcloud2, interpolation='bilinear')
plt.title("Word Cloud (After Preprocessing)")
plt.axis("off")
plt.show()

# Task 5
# Task 5: Manual POS tagging of 3 sentences
sentences = [
    "The cat sat on the mat.",
    "John plays football every Sunday.",
    "Artificial intelligence is the future."
]

for s in sentences:
    words = [w.strip('.,').lower() for w in s.split()]
    tags = []
    for w in words:
        if w.endswith("ing") or w in {"plays", "sat", "is"}:
            tag = "VERB"
        elif w in {"the", "a", "on"}:
            tag = "DET/PREP"
        elif w in {"john", "cat", "mat", "football", "intelligence", "future"}:
            tag = "NOUN"
        else:
            tag = "OTHER"
        tags.append((w, tag))
    print(tags)

# Task 6
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

sentences = [
    "The cat sat on the mat.",
    "John plays football every Sunday.",
    "Artificial intelligence is the future."
]

for s in sentences:
    tokens = word_tokenize(s)
    pos_tags = pos_tag(tokens)
    print(pos_tags)

# Task 7
import spacy

nlp = spacy.load("en_core_web_sm")

sentences = [
    "The cat sat on the mat.",
    "John plays football every Sunday.",
    "Artificial intelligence is the future."
]

for s in sentences:
    doc = nlp(s)
    tags = [(token.text, token.pos_) for token in doc]
    print(tags)

# Task 8
import nltk
import spacy
import requests
import time
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nlp = spacy.load("en_core_web_sm")

url = "https://shorturl.at/WeE7u"
text = requests.get(url).text

words = [w for w in word_tokenize(text.lower()) if w.isalpha()]

start = time.time()
pos_nltk = nltk.pos_tag(words)
end = time.time()
print(f"NLTK tagging time: {end - start:.2f} seconds")

start = time.time()
doc = nlp(" ".join(words))
pos_spacy = [(token.text, token.pos_) for token in doc]
end = time.time()
print(f"spaCy tagging time: {end - start:.2f} seconds")