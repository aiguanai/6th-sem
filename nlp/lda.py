"""
Program No. 7 - Topic Modeling using TF-IDF Vectorization and LDA Algorithm
Dataset: PDF Documents
Concept: Topic modeling using TF-IDF Vectorization and LDA Algorithm
"""

import os
import re
import math
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from PyPDF2 import PdfReader

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Stopwords
STOP_WORDS = set(stopwords.words('english'))

# ─────────────────────────────────────────────
# STEP 1: LOAD PDF DOCUMENTS
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_path):

    text = ""

    try:

        reader = PdfReader(pdf_path)

        for page in reader.pages:

            content = page.extract_text()

            if content:
                text += content + " "

    except Exception as e:

        print(f"Error reading {pdf_path}: {e}")

    return text


def load_pdfs(folder_path):

    documents = []
    filenames = []

    for file in os.listdir(folder_path):

        if file.endswith(".pdf"):

            full_path = os.path.join(folder_path, file)

            text = extract_text_from_pdf(full_path)

            if text.strip():

                documents.append(text)
                filenames.append(file)

    return documents, filenames


# ─────────────────────────────────────────────
# STEP 2: PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    words = word_tokenize(text)

    words = [
        w for w in words
        if w not in STOP_WORDS and len(w) > 2
    ]

    return words


# ─────────────────────────────────────────────
# STEP 3: BUILD VOCABULARY
# ─────────────────────────────────────────────

def build_vocabulary(docs):

    doc_freq = defaultdict(int)

    for doc in docs:

        for word in set(doc):
            doc_freq[word] += 1

    N = len(docs)

    idf = {}

    for word, freq in doc_freq.items():

        df_ratio = freq / N

        if freq >= 2 and df_ratio <= 0.85:

            idf[word] = math.log((N + 1) / (freq + 1)) + 1

    sorted_vocab = sorted(
        idf.keys(),
        key=lambda x: idf[x],
        reverse=True
    )[:1000]

    vocab = {
        word: idx
        for idx, word in enumerate(sorted_vocab)
    }

    return vocab, idf


# ─────────────────────────────────────────────
# STEP 4: TF-IDF VECTORIZATION
# ─────────────────────────────────────────────

def compute_tfidf(processed_docs, vocab, idf):

    N = len(processed_docs)
    V = len(vocab)

    tfidf_matrix = np.zeros((N, V))

    for d, doc in enumerate(processed_docs):

        tf = defaultdict(int)

        for word in doc:
            tf[word] += 1

        total_words = len(doc)

        if total_words == 0:
            continue

        for word in tf:
            tf[word] = tf[word] / total_words

        for word in tf:

            if word in vocab and word in idf:

                idx = vocab[word]

                tfidf_matrix[d][idx] = tf[word] * idf[word]

    return tfidf_matrix


# ─────────────────────────────────────────────
# STEP 5: LDA MODEL
# ─────────────────────────────────────────────

class LDA:

    def __init__(
        self,
        n_topics=3,
        n_iter=100,
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

        docs = []

        for doc in docs_tokens:

            filtered = [
                vocab[word]
                for word in doc
                if word in vocab
            ]

            docs.append(filtered)

        n_dk = np.zeros((D, K), dtype=int)
        n_kw = np.zeros((K, V), dtype=int)
        n_k = np.zeros(K, dtype=int)
        n_d = np.array([len(doc) for doc in docs])

        z = []

        for d, doc in enumerate(docs):

            z_d = []

            for w in doc:

                topic = random.randint(0, K - 1)

                z_d.append(topic)

                n_dk[d][topic] += 1
                n_kw[topic][w] += 1
                n_k[topic] += 1

            z.append(z_d)

        print(f"\nTraining LDA ({K} topics)...")

        # Gibbs Sampling
        for iteration in range(self.n_iter):

            for d, doc in enumerate(docs):

                for i, w in enumerate(doc):

                    old_topic = z[d][i]

                    n_dk[d][old_topic] -= 1
                    n_kw[old_topic][w] -= 1
                    n_k[old_topic] -= 1

                    probs = np.zeros(K)

                    for k in range(K):

                        probs[k] = (
                            (n_dk[d][k] + self.alpha)
                            *
                            (n_kw[k][w] + self.beta)
                            /
                            (n_k[k] + V * self.beta)
                        )

                    # Safe normalization
                    if probs.sum() == 0:
                        probs = np.ones(K) / K
                    else:
                        probs = probs / probs.sum()

                    new_topic = np.random.choice(K, p=probs)

                    z[d][i] = new_topic

                    n_dk[d][new_topic] += 1
                    n_kw[new_topic][w] += 1
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
# EXECUTION
# ─────────────────────────────────────────────

folder_path = input("Enter PDF folder path: ")

num_topics = int(input("Enter number of topics: "))

# Load PDFs
documents, filenames = load_pdfs(folder_path)

if len(documents) == 0:

    print("No PDF files found!")

    exit()

print(f"\nLoaded {len(documents)} PDF documents.")

# Preprocess documents
processed_docs = [
    preprocess(doc)
    for doc in documents
]

# Remove empty documents
processed_docs = [
    doc for doc in processed_docs
    if len(doc) > 0
]

print("Preprocessing complete.")

# Build vocabulary
vocab, idf = build_vocabulary(processed_docs)

print("Vocabulary Size:", len(vocab))

# TF-IDF Vectorization
tfidf_matrix = compute_tfidf(
    processed_docs,
    vocab,
    idf
)

print("TF-IDF Vectorization complete.")

# Train LDA
lda = LDA(
    n_topics=num_topics,
    n_iter=100,
    alpha=0.1,
    beta=0.01
)

lda.fit(processed_docs, vocab)

# Top words
print("\n" + "=" * 60)
print("TOP WORDS PER TOPIC")
print("=" * 60)

top_words = lda.get_top_words(n_top=10)

for k, words in enumerate(top_words):

    print(f"\nTopic {k+1}:")
    print(", ".join(words))

# Document-topic assignment
doc_topics = lda.get_doc_topics()

topic_to_files = {
    i: []
    for i in range(num_topics)
}

for i, topic in enumerate(doc_topics):

    topic_to_files[topic].append(filenames[i])

# Display document clusters
print("\n" + "=" * 60)
print("DOCUMENT CLUSTERS")
print("=" * 60)

for t in range(num_topics):

    print(f"\nTopic {t+1}:")

    if len(topic_to_files[t]) == 0:

        print("No PDFs assigned")

    else:

        for file in topic_to_files[t]:

            print("-", file)

# Save results
results = []

for i, topic in enumerate(doc_topics):

    results.append([
        filenames[i],
        topic + 1
    ])

results_df = pd.DataFrame(
    results,
    columns=["PDF_File", "Assigned_Topic"]
)

results_df.to_csv(
    "pdf_topic_results.csv",
    index=False
)

print("\nResults saved to 'pdf_topic_results.csv'")
