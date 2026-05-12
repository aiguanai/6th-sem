"""
Program No. 6 — Email Spam Detection
TF-IDF Vectorizer (hardcoded) + Random Forest (hardcoded)
Each tree in the forest uses sklearn's DecisionTreeClassifier.
Dataset: spam.csv
"""

import pandas as pd
import numpy as np
import math
import re
import nltk
import random
from collections import Counter
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from sklearn.tree import DecisionTreeClassifier

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

# ─────────────────────────────────────────────────────────────
# STEP 1 — Load Data
# ─────────────────────────────────────────────────────────────

df = pd.read_csv('spam.csv', encoding='latin-1', usecols=[0, 1])
df.columns = ['label', 'text']
df.dropna(inplace=True)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

print(f"Dataset loaded : {len(df)} samples")
print(f"  Spam : {df['label_num'].sum()}")
print(f"  Ham  : {(df['label_num'] == 0).sum()}")

# ─────────────────────────────────────────────────────────────
# STEP 2 — Preprocessing
# ─────────────────────────────────────────────────────────────

STOP_WORDS = set(stopwords.words('english'))

def preprocess(text):
    """Remove HTML tags, lowercase, keep letters only, remove stopwords."""
    text = re.sub(r'<[^>]+>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

print("\nPreprocessing emails ...")
df['tokens'] = df['text'].apply(preprocess)

# ─────────────────────────────────────────────────────────────
# STEP 3 — Hardcoded TF-IDF Vectorizer
# ─────────────────────────────────────────────────────────────

class TFIDFVectorizer:
    """
    Hardcoded TF-IDF with Binary-Count encoding.

    Feature encoding:
        feature(w, d) = 1.0   if word w present in document d
                      = 0.0   if word w absent
        Spam emails typically hit 5-15 vocab words; ham hits 0-1.

    IDF(w) = log( (1+N) / (1+df_w) ) + 1   [smoothed, for vocab selection only]

    Vocabulary selection — score(w) = chi2(w) × precision(w):
        chi2(w)      = χ² statistic of word-spam association
        precision(w) = spam_docs_with_w / all_docs_with_w
        Words with precision < min_precision are excluded
        (e.g. 'call' prec=60%, appears heavily in ham → false positives).
    """

    def __init__(self, max_features=500, min_precision=0.65):
        self.max_features  = max_features
        self.min_precision = min_precision
        self.vocabulary_   = {}   # word -> column index
        self.index_word_   = []   # index -> word

    def _compute_idf(self, token_lists):
        N        = len(token_lists)
        df_count = Counter()
        for tokens in token_lists:
            for w in set(tokens):
                df_count[w] += 1
        return {w: math.log((1 + N) / (1 + df)) + 1
                for w, df in df_count.items() if df >= 2}

    def _compute_chi2_precision(self, token_lists, labels):
        n_spam = sum(1 for l in labels if l == 1)
        n_ham  = sum(1 for l in labels if l == 0)
        N      = len(labels)

        word_in_spam = Counter()
        word_in_ham  = Counter()
        for tokens, lbl in zip(token_lists, labels):
            for w in set(tokens):
                if lbl == 1: word_in_spam[w] += 1
                else:        word_in_ham[w]  += 1

        chi2_scores = {}
        precision   = {}
        for w in set(word_in_spam) | set(word_in_ham):
            A = word_in_spam.get(w, 0)
            B = word_in_ham.get(w, 0)
            if A + B < 3:
                continue
            C     = n_spam - A
            D     = n_ham  - B
            denom = (A + B) * (C + D) * (A + C) * (B + D)
            if denom > 0:
                chi2_scores[w] = N * (A * D - B * C) ** 2 / denom
            precision[w] = A / (A + B)

        return chi2_scores, precision

    def fit(self, token_lists, labels):
        print("  Computing chi2 × precision scores for vocabulary selection ...")
        idf_raw           = self._compute_idf(token_lists)
        chi2_scores, prec = self._compute_chi2_precision(token_lists, labels)

        word_score = {
            w: chi2_scores[w] * prec[w]
            for w in chi2_scores
            if prec.get(w, 0) >= self.min_precision and w in idf_raw
        }

        top_words        = sorted(word_score, key=lambda w: word_score[w], reverse=True)
        top_words        = top_words[:self.max_features]
        self.index_word_ = top_words
        self.vocabulary_ = {w: i for i, w in enumerate(top_words)}
        return self

    def _vec(self, tokens):
        vec = [0.0] * len(self.vocabulary_)
        for w in set(tokens):
            if w in self.vocabulary_:
                vec[self.vocabulary_[w]] = 1.0
        return vec

    def transform(self, token_lists):
        return np.array([self._vec(t) for t in token_lists])

    def fit_transform(self, token_lists, labels):
        self.fit(token_lists, labels)
        return self.transform(token_lists)

# ─────────────────────────────────────────────────────────────
# STEP 4 — Fit Vectorizer and Train/Test Split
# ─────────────────────────────────────────────────────────────

print("\nFitting TF-IDF vectorizer ...")
tfidf = TFIDFVectorizer(max_features=500, min_precision=0.65)
X     = tfidf.fit_transform(df['tokens'].tolist(), df['label_num'].tolist())
y     = df['label_num'].values

print(f"\nTop spam-signal words in vocabulary:")
print(f"  {tfidf.index_word_[:25]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain : {len(X_train)}  (spam={y_train.sum()})")
print(f"Test  : {len(X_test)}   (spam={y_test.sum()})")

spam_hits = X_test[y_test == 1].sum(1)
ham_hits  = X_test[y_test == 0].sum(1)
print(f"\nVocab hits per email:")
print(f"  Spam — min:{int(spam_hits.min())}  median:{int(np.median(spam_hits))}  max:{int(spam_hits.max())}")
print(f"  Ham  — min:{int(ham_hits.min())}   p90:{int(np.percentile(ham_hits,90))}   max:{int(ham_hits.max())}")
print(f"Spam emails with ≥1 vocab hit : {(spam_hits >= 1).mean():.1%}")

# ─────────────────────────────────────────────────────────────
# STEP 4b — Oversample minority class on training set
# ─────────────────────────────────────────────────────────────

def oversample_minority(X, y, seed=42):
    """Duplicate spam rows with 5% random feature dropout for diversity."""
    random.seed(seed)
    np.random.seed(seed)
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    if n1 >= n0:
        return X, y
    needed  = n0 - n1
    src_idx = np.where(y == 1)[0]
    chosen  = np.random.choice(src_idx, needed, replace=True)
    X_extra = X[chosen].copy()
    drop    = np.random.random(X_extra.shape) < 0.05
    X_extra[drop] = 0.0
    X_new   = np.vstack([X, X_extra])
    y_new   = np.concatenate([y, np.ones(needed, dtype=y.dtype)])
    shuf    = np.random.permutation(len(y_new))
    return X_new[shuf], y_new[shuf]

X_bal, y_bal = oversample_minority(X_train, y_train)
print(f"\nAfter oversampling — Ham: {int((y_bal==0).sum())}  Spam: {int((y_bal==1).sum())}")

# ─────────────────────────────────────────────────────────────
# STEP 5 — Hardcoded Random Forest
#           (each tree = sklearn DecisionTreeClassifier)
# ─────────────────────────────────────────────────────────────

class RandomForest:
    """
    Hardcoded Random Forest using sklearn DecisionTreeClassifier per tree.

    Algorithm:
        1. For each of n_trees iterations:
           a. Draw a bootstrap sample (random sample WITH replacement)
              of size n from the training set.
           b. Train a DecisionTreeClassifier on the bootstrap sample
              with max_features=sqrt(n_features) so each split only
              considers a random subset of features (key RF property).
        2. Prediction via Majority Voting:
              spam_fraction = trees_voting_spam / n_trees
              predict SPAM  if spam_fraction >= vote_threshold
              predict HAM   otherwise
        vote_threshold=0.45 is slightly below 0.5 to improve recall
        on borderline spam without sacrificing much precision.
    """

    def __init__(self, n_trees=40, max_depth=20, min_samples_split=2,
                 max_features='sqrt', class_weight='balanced',
                 vote_threshold=0.45, seed=42):
        self.n_trees           = n_trees
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.class_weight      = class_weight
        self.vote_threshold    = vote_threshold
        self.seed              = seed
        self.trees_            = []

    def fit(self, X, y):
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.trees_ = []
        n = X.shape[0]

        print(f"\nTraining Random Forest ({self.n_trees} trees, "
              f"max_depth={self.max_depth}, "
              f"vote_threshold={self.vote_threshold}) ...")

        for i in range(self.n_trees):
            # Step 1a: Bootstrap sample (with replacement)
            boot_idx = np.random.choice(n, n, replace=True)
            X_boot   = X[boot_idx]
            y_boot   = y[boot_idx]

            # Step 1b: Train sklearn DecisionTree on bootstrap sample
            tree = DecisionTreeClassifier(
                max_depth         = self.max_depth,
                min_samples_split = self.min_samples_split,
                max_features      = self.max_features,   # sqrt at each split
                class_weight      = self.class_weight,
                random_state      = self.seed + i,
            )
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)
            print(f"  Tree {i+1:>2}/{self.n_trees} done", end='\r')

        print(f"\nAll {self.n_trees} trees trained.")
        return self

    def predict(self, X):
        """
        Majority Voting across all trees.
        votes shape : (n_trees, n_samples)
        spam_frac   : fraction of trees voting SPAM per sample
        """
        votes     = np.vstack([tree.predict(X) for tree in self.trees_])
        spam_frac = votes.mean(axis=0)
        return (spam_frac >= self.vote_threshold).astype(int)

# ─────────────────────────────────────────────────────────────
# STEP 6 — Train
# ─────────────────────────────────────────────────────────────

rf = RandomForest(
    n_trees           = 40,
    max_depth         = 20,
    min_samples_split = 2,
    max_features      = 'sqrt',
    class_weight      = 'balanced',
    vote_threshold    = 0.45,
    seed              = 42,
)
rf.fit(X_bal, y_bal)

# ─────────────────────────────────────────────────────────────
# STEP 7 — Evaluation Metrics (sklearn)
# ─────────────────────────────────────────────────────────────

print("\nEvaluating on held-out test set ...")
y_pred = rf.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

W = 54
print("\n" + "=" * W)
print("           MODEL EVALUATION RESULTS")
print("=" * W)
print(f"  Accuracy   : {acc  * 100:.2f}%")
print(f"  Precision  : {prec * 100:.2f}%")
print(f"  Recall     : {rec  * 100:.2f}%")
print(f"  F1-Score   : {f1   * 100:.2f}%")
print()
print("  Confusion Matrix:")
print(f"  {'':18s}  Pred Ham   Pred Spam")
print(f"  {'Actual Ham':18s}  {tn:<10} {fp}")
print(f"  {'Actual Spam':18s}  {fn:<10} {tp}")
print()
print("  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
print("=" * W)

# ─────────────────────────────────────────────────────────────
# STEP 8 — Predict New Emails
# ─────────────────────────────────────────────────────────────

def predict_email(raw_text):
    tokens = preprocess(raw_text)
    vec    = tfidf.transform([tokens])
    return "SPAM" if rf.predict(vec)[0] == 1 else "HAM"

print("\n" + "=" * W)
print("           SAMPLE PREDICTIONS")
print("=" * W)

examples = [
    "URGENT! You have won a 1 week FREE membership in our 100,000 Prize Jackpot!",
    "Did you catch the bus?",
    "Congratulations! You've been selected for a free iPhone. Click now to claim!",
    "Hey, are we meeting for lunch tomorrow?",
    "FREE entry! Win cash prizes worth $1000. Text WIN to 80082 now!",
    "Call me when you get home, dinner is ready.",
    "You have been awarded a $500 Walmart gift card. Claim it now!",
    "Reminder: your appointment is tomorrow at 10am.",
]

for email in examples:
    result = predict_email(email)
    tag    = "SPAM" if result == "SPAM" else "HAM "
    print(f"  [{tag}]  {email[:62]}")
print()
