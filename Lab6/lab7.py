# lab 7 task 1
def ngram_calc(filepath, n=2):

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    text = text.lower()
    token = text.split()
    ngrams = [tuple(token[i:i+n]) 
              for i in range(len(token)-n+1)]

    return ngrams


ngrams = ngram_calc("Lab6/input.txt", n=3)
print(ngrams[:20])

# task 2
import math
from collections import Counter

def load_tokens(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().lower()
    return text.split()

def ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def perplexity_unigram(unigram_counts, total_tokens):
    N = total_tokens
    log_sum = 0

    for word, count in unigram_counts.items():
        prob = count / N
        log_sum += count * math.log(prob)

    return math.exp(-log_sum / N)

def perplexity_bigram(unigram_counts, bigram_counts):
    N = sum(bigram_counts.values())
    log_sum = 0

    for bigram, count in bigram_counts.items():
        w1 = (bigram[0],)
        prob = count / unigram_counts[w1]   # P(w2 | w1)

        log_sum += count * math.log(prob)

    return math.exp(-log_sum / N)


tokens = load_tokens("Lab6/input.txt")

unigram_counts = ngram_counts(tokens, 1)
bigram_counts  = ngram_counts(tokens, 2)

print("Unigram Perplexity:", perplexity_unigram(unigram_counts, len(tokens)))
print("Bigram  Perplexity:", perplexity_bigram(unigram_counts, bigram_counts))


#  task 3
def laplace_unigram_prob(unigram_counts, V):
    N = sum(unigram_counts.values())
    probs = {}

    for word in unigram_counts:
        probs[word] = (unigram_counts[word] + 1) / (N + V)

    return probs


def laplace_bigram_prob(unigram_counts, bigram_counts, V):
    probs = {}

    for (w1, w2), count in bigram_counts.items():
        probs[(w1, w2)] = (count + 1) / (unigram_counts[(w1,)] + V)

    return probs



V = len(unigram_counts)

laplace_uni = laplace_unigram_prob(unigram_counts, V)
laplace_bi  = laplace_bigram_prob(unigram_counts, bigram_counts, V)

print("Sample Laplace Unigram:", list(laplace_uni.items())[:10])
print("Sample Laplace Bigram:", list(laplace_bi.items())[:10])
