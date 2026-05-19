"""
Program No. 7 - Topic Modeling using TF-IDF Vectorization and LDA Algorithm
Dataset: Quora Questions
Concept: Topic modeling using TF-IDF Vectorization and LDA Algorithm
Using NumPy and Pandas
"""

import math
import random
import pandas as pd
import numpy as np
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ─────────────────────────────────────────────
# STEP 1: Load & Preprocess Data
# ─────────────────────────────────────────────

# Load dataset using pandas
df = pd.read_csv("/content/quora_questions (1).csv")

# Take first 500 questions
documents = df.iloc[:500, 0].dropna().astype(str).tolist()

print(f"Loaded {len(documents)} documents.")

# Stopwords
stop_words = set(stopwords.words('english'))

# Text preprocessing
processed = []

for doc in documents:

    tokens = word_tokenize(doc.lower())

    # Remove stopwords, punctuation, short words
    tokens = [
        token for token in tokens
        if token.isalpha()
        and token not in stop_words
        and len(token) > 2
    ]

    processed.append(tokens)

# FIX 5: Remove empty documents
processed = [doc for doc in processed if len(doc) > 0]

print("Preprocessing complete.")

# ─────────────────────────────────────────────
# STEP 2: TF-IDF Vectorization using NumPy
# ─────────────────────────────────────────────

# Build vocabulary
all_words = []

for doc in processed:
    all_words.extend(doc)

unique_words = list(set(all_words))

# Document Frequency (DF)
doc_freq = defaultdict(int)

for doc in processed:
    for word in set(doc):
        doc_freq[word] += 1

N = len(processed)

# Compute IDF
idf = {}

for word, freq in doc_freq.items():

    df_ratio = freq / N

    if freq >= 2 and df_ratio <= 0.85:
        idf[word] = math.log((N + 1) / (freq + 1)) + 1

# Select top 1000 words
sorted_vocab = sorted(
    idf.keys(),
    key=lambda x: idf[x],
    reverse=True
)[:1000]

vocab = {word: idx for idx, word in enumerate(sorted_vocab)}
idx_to_word = {idx: word for word, idx in vocab.items()}

V = len(vocab)

print(f"Vocabulary size: {V}")

# TF-IDF Matrix
tfidf_matrix = np.zeros((N, V))

for d, doc in enumerate(processed):

    # Term Frequency
    tf = defaultdict(int)

    for word in doc:
        tf[word] += 1

    total_words = len(doc)

    if total_words == 0:
        continue

    for word in tf:
        tf[word] = tf[word] / total_words

    # FIX 1: Use tf instead of doc
    for word in tf:

        if word in vocab and word in idf:

            w_idx = vocab[word]

            tfidf_matrix[d][w_idx] = tf[word] * idf[word]

print("TF-IDF Vectorization complete.")

# ─────────────────────────────────────────────
# STEP 3 & 4: LDA using Gibbs Sampling
# ─────────────────────────────────────────────

class LDA:

    def __init__(
        self,
        n_topics=5,
        n_iter=50,
        alpha=0.1,
        beta=0.01,
        random_state=42
    ):

        self.K = n_topics
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta

        random.seed(random_state)
        np.random.seed(random_state)

    def fit(self, docs_tokens, vocab):

        self.vocab = vocab

        self.idx_to_word = {
            v: k for k, v in vocab.items()
        }

        D = len(docs_tokens)
        V = len(vocab)
        K = self.K

        # Convert tokens to indices
        docs = []

        for doc in docs_tokens:

            filtered = [
                vocab[word]
                for word in doc
                if word in vocab
            ]

            docs.append(filtered)

        # Count matrices
        n_dk = np.zeros((D, K), dtype=int)
        n_kw = np.zeros((K, V), dtype=int)
        n_k = np.zeros(K, dtype=int)
        n_d = np.array([len(doc) for doc in docs])

        # Random topic assignments
        z = []

        for d, doc in enumerate(docs):

            z_d = []

            for w in doc:

                topic = random.randint(0, K - 1)

                z_d.append(topic)

                n_dk[d, topic] += 1
                n_kw[topic, w] += 1
                n_k[topic] += 1

            z.append(z_d)

        print(f"\nTraining LDA ({K} topics, {self.n_iter} iterations)...")

        # Gibbs Sampling
        for iteration in range(self.n_iter):

            for d, doc in enumerate(docs):

                for i, w in enumerate(doc):

                    old_topic = z[d][i]

                    # Remove old assignment
                    n_dk[d, old_topic] -= 1
                    n_kw[old_topic, w] -= 1
                    n_k[old_topic] -= 1

                    # Compute topic probabilities
                    probs = np.zeros(K)

                    for k in range(K):

                        probs[k] = (
                            (n_dk[d, k] + self.alpha)
                            *
                            (n_kw[k, w] + self.beta)
                            /
                            (n_k[k] + V * self.beta)
                        )

                    # FIX 2: Safe normalization
                    if probs.sum() == 0:
                        probs = np.ones(K) / K
                    else:
                        probs = probs / probs.sum()

                    # Sample new topic
                    new_topic = np.random.choice(K, p=probs)

                    # Reassign topic
                    z[d][i] = new_topic

                    n_dk[d, new_topic] += 1
                    n_kw[new_topic, w] += 1
                    n_k[new_topic] += 1

            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iter} complete")

        # Topic-word distribution
        self.phi = np.zeros((K, V))

        for k in range(K):

            self.phi[k] = (
                n_kw[k] + self.beta
            ) / (
                n_k[k] + V * self.beta
            )

        # Document-topic distribution
        self.theta = np.zeros((D, K))

        for d in range(D):

            self.theta[d] = (
                n_dk[d] + self.alpha
            ) / (
                n_d[d] + K * self.alpha
            )

    def get_top_words(self, n_top=10):

        top_words = []

        for k in range(self.K):

            top_indices = np.argsort(self.phi[k])[::-1][:n_top]

            words = [
                self.idx_to_word[i]
                for i in top_indices
            ]

            top_words.append(words)

        return top_words

    def get_doc_topics(self):

        return np.argmax(self.theta, axis=1)

# ─────────────────────────────────────────────
# Train LDA Model
# ─────────────────────────────────────────────

lda = LDA(
    n_topics=5,
    n_iter=50,
    alpha=0.1,
    beta=0.01,
    random_state=42
)

lda.fit(processed, vocab)

# ─────────────────────────────────────────────
# STEP 5: Top Words per Topic
# ─────────────────────────────────────────────

top_words = lda.get_top_words(n_top=5)

# ─────────────────────────────────────────────
# STEP 6: Topic Assignments
# ─────────────────────────────────────────────

doc_topics = lda.get_doc_topics()

# ─────────────────────────────────────────────
# STEP 7: Result Visualization
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("        TOPIC MODELING RESULTS (LDA)")
print("=" * 60)

# Top Words
print("\n--- Top Words per Topic ---")

for k, words in enumerate(top_words):

    print(f"\nTopic {k+1}: {', '.join(words)}")

# Document Topic Assignments
print("\n--- Document-Topic Assignments (Sample) ---")

print(f"{'#':<5} {'Topic':<8} {'Question'}")
print("-" * 80)

for i in range(min(25, len(documents))):

    topic = doc_topics[i] + 1

    question = (
        documents[i][:70] + "..."
        if len(documents[i]) > 70
        else documents[i]
    )

    print(f"{i+1:<5} Topic {topic:<3} {question}")

# Topic Distribution
print("\n--- Topic Distribution Across Documents ---")

topic_counts = defaultdict(int)

for topic in doc_topics:
    topic_counts[topic + 1] += 1

for k in sorted(topic_counts):

    count = topic_counts[k]

    percentage = (count / N) * 100

    bar = "█" * int((count / N) * 40)

    print(f"Topic {k}: {bar} ({count} docs, {percentage:.1f}%)")

print("\n" + "=" * 60)

# ─────────────────────────────────────────────
# OPTIONAL: Save Results
# ─────────────────────────────────────────────

results_df = pd.DataFrame({
    "Question": documents[:len(doc_topics)],
    "Assigned_Topic": doc_topics + 1
})

results_df.to_csv("lda_topic_results.csv", index=False)

print("\nResults saved to 'lda_topic_results.csv'")
