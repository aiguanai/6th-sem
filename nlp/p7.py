"""
Program No. 7 - Topic Modeling using TF-IDF Vectorization and LDA Algorithm
Dataset: Quora Questions
Concept: Topic modeling using the vectorization method and LDA algorithm.
"""

import math
import random
import csv
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ─────────────────────────────────────────────
# STEP 1: Load & Preprocess Data
# ─────────────────────────────────────────────

documents = []
with open("/content/quora_questions (1).csv", newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for i, row in enumerate(reader):
        if i >= 500:
            break
        if row:
            documents.append(row[0].strip())

print(f"Loaded {len(documents)} documents.")

stop_words = set(stopwords.words('english'))
processed = []
for doc in documents:
    tokens = word_tokenize(doc.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 2]
    processed.append(tokens)

print("Preprocessing complete.")

# ─────────────────────────────────────────────
# STEP 2: Hardcoded TF-IDF Vectorization
# ─────────────────────────────────────────────

# Compute TF
tf_all = []
for doc_tokens in processed:
    tf = {}
    total = len(doc_tokens)
    if total == 0:
        tf_all.append(tf)
        continue
    for token in doc_tokens:
        tf[token] = tf.get(token, 0) + 1
    for token in tf:
        tf[token] = tf[token] / total
    tf_all.append(tf)

# Compute IDF
N = len(processed)
doc_freq = defaultdict(int)
for doc in processed:
    for token in set(doc):
        doc_freq[token] += 1

idf = {}
for token, freq in doc_freq.items():
    df_ratio = freq / N
    if freq >= 2 and df_ratio <= 0.85:
        idf[token] = math.log((N + 1) / (freq + 1)) + 1  # smoothed IDF

# Select top 1000 terms by IDF score
sorted_vocab = sorted(idf.keys(), key=lambda t: idf[t], reverse=True)[:1000]
vocab = {term: idx for idx, term in enumerate(sorted_vocab)}
idx_to_word = {v: k for k, v in vocab.items()}

# Build TF-IDF matrix
tfidf_matrix = []
for i, doc_tokens in enumerate(processed):
    row = {}
    for token in doc_tokens:
        if token in vocab and token in idf:
            row[vocab[token]] = tf_all[i].get(token, 0) * idf[token]
    tfidf_matrix.append(row)

print(f"Vocabulary size: {len(vocab)} terms")

# ─────────────────────────────────────────────
# STEP 3 & 4: LDA Class (Collapsed Gibbs Sampling)
# ─────────────────────────────────────────────

class LDA:
    def __init__(self, n_topics=5, n_iter=50, alpha=0.1, beta=0.01, random_state=42):
        self.K = n_topics
        self.n_iter = n_iter
        self.alpha = alpha   # Dirichlet prior on doc-topic distribution
        self.beta = beta     # Dirichlet prior on topic-word distribution
        random.seed(random_state)

    def fit(self, docs_tokens, vocab):
        self.vocab = vocab
        self.idx_to_word = {v: k for k, v in vocab.items()}
        V = len(vocab)
        K = self.K
        D = len(docs_tokens)

        # Convert token strings to word indices
        docs = []
        for doc in docs_tokens:
            filtered = [vocab[t] for t in doc if t in vocab]
            docs.append(filtered)

        # Count matrices
        n_dk = [[0] * K for _ in range(D)]  # doc-topic counts
        n_kw = [[0] * V for _ in range(K)]  # topic-word counts
        n_k  = [0] * K                       # total words per topic
        n_d  = [len(doc) for doc in docs]    # total words per doc

        # Random initial topic assignments
        z = []
        for d, doc in enumerate(docs):
            z_d = []
            for w in doc:
                k = random.randint(0, K - 1)
                z_d.append(k)
                n_dk[d][k] += 1
                n_kw[k][w] += 1
                n_k[k] += 1
            z.append(z_d)

        # Gibbs Sampling iterations
        print(f"\nTraining LDA ({K} topics, {self.n_iter} iterations)...")
        for iteration in range(self.n_iter):
            for d, doc in enumerate(docs):
                for i, w in enumerate(doc):
                    k_old = z[d][i]

                    # Remove current word's assignment
                    n_dk[d][k_old] -= 1
                    n_kw[k_old][w] -= 1
                    n_k[k_old] -= 1

                    # Compute probability for each topic
                    probs = []
                    for k in range(K):
                        p = ((n_dk[d][k] + self.alpha) *
                             (n_kw[k][w] + self.beta) /
                             (n_k[k] + V * self.beta))
                        probs.append(p)

                    # Normalize
                    total_p = sum(probs)
                    probs = [p / total_p for p in probs]

                    # Sample new topic
                    r = random.random()
                    cumulative = 0.0
                    k_new = K - 1
                    for k, p in enumerate(probs):
                        cumulative += p
                        if r <= cumulative:
                            k_new = k
                            break

                    # Reassign
                    z[d][i] = k_new
                    n_dk[d][k_new] += 1
                    n_kw[k_new][w] += 1
                    n_k[k_new] += 1

            if (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration + 1}/{self.n_iter} complete")

        # Phi: topic-word distribution P(w|k)
        self.phi = []
        for k in range(K):
            phi_k = [(n_kw[k][w] + self.beta) / (n_k[k] + V * self.beta)
                     for w in range(V)]
            self.phi.append(phi_k)

        # Theta: document-topic distribution P(k|d)
        self.theta = []
        for d in range(D):
            theta_d = [(n_dk[d][k] + self.alpha) / (n_d[d] + K * self.alpha)
                       for k in range(K)]
            self.theta.append(theta_d)

    def get_top_words(self, n_top=10):
        top_words = []
        for k in range(self.K):
            sorted_words = sorted(range(len(self.phi[k])),
                                  key=lambda w: self.phi[k][w], reverse=True)
            words = [self.idx_to_word[w] for w in sorted_words[:n_top]]
            top_words.append(words)
        return top_words

    def get_doc_topics(self):
        return [max(range(self.K), key=lambda k: theta[k])
                for theta in self.theta]


# ─────────────────────────────────────────────
# Train LDA
# ─────────────────────────────────────────────

lda = LDA(n_topics=5, n_iter=50, alpha=0.1, beta=0.01, random_state=42)
lda.fit(processed, vocab)

# ─────────────────────────────────────────────
# STEP 5: Top Words per Topic
# ─────────────────────────────────────────────

top_words = lda.get_top_words(n_top=3)

# ─────────────────────────────────────────────
# STEP 6: Topic Assignments
# ─────────────────────────────────────────────

doc_topics = lda.get_doc_topics()

# ─────────────────────────────────────────────
# STEP 7: Result Visualization
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("         TOPIC MODELING RESULTS (LDA)")
print("="*60)

print("\n--- Top Words per Topic ---")
for k, words in enumerate(top_words):
    print(f"\nTopic {k+1}: {', '.join(words)}")

print("\n\n--- Document-Topic Assignments (sample) ---")
print(f"{'#':<5} {'Topic':<8} {'Question (truncated)'}")
print("-" * 70)
for i in range(25):
    topic = doc_topics[i] + 1
    question = documents[i][:70] + "..." if len(documents[i]) > 70 else documents[i]
    print(f"{i+1:<5} Topic {topic:<3} {question}")

print("\n\n--- Topic Distribution across Documents ---")
topic_counts = defaultdict(int)
for t in doc_topics:
    topic_counts[t + 1] += 1

D = len(documents)
for k in sorted(topic_counts):
    count = topic_counts[k]
    bar = "█" * (count * 40 // D)
    print(f"Topic {k}: {bar} ({count} docs, {100*count/D:.1f}%)")

print("\n" + "="*60)
