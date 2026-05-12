"""
Program No. 6 — Email Spam Detection
Using hardcoded TF-IDF Vectorizer and hardcoded Random Forest algorithm
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
    """Remove HTML tags, lowercase, keep letters, remove stopwords."""
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
    Hardcoded TF-IDF with Binary-Count encoding and chi2×precision vocabulary.

    Feature encoding:
        feature(w, d) = 1.0   if word w present in document d
                      = 0.0   if word w absent  (exact zero)
        No TF, no IDF weighting on features, no L2 norm.
        Spam emails hit 5-15 vocab words; ham emails hit 0-1.
        Threshold = 0.0 cleanly splits word-absent vs word-present.

    IDF(w) = log( (1+N) / (1+df_w) ) + 1   [smoothed, used for vocab selection only]

    Vocabulary selection:
        chi2(w)      = χ² statistic of word-spam association
        precision(w) = spam_docs_with_w / all_docs_with_w
        score(w)     = chi2(w) × precision(w)   if precision >= min_precision
        Top max_features words by this score form the vocabulary.
        min_precision filter removes ambiguous words like 'call' (prec=60%)
        that appear in both spam and ham, causing false positives.
    """

    def __init__(self, max_features=500, min_precision=0.65):
        self.max_features  = max_features
        self.min_precision = min_precision
        self.vocabulary_   = {}   # word -> column index
        self.index_word_   = []   # index -> word (for display)

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
        idf_raw              = self._compute_idf(token_lists)
        chi2_scores, prec    = self._compute_chi2_precision(token_lists, labels)

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
        """Binary count: 1.0 if vocab word present, exact 0.0 if absent."""
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
# STEP 4 — Fit vectorizer and Train/Test Split (sklearn)
# ─────────────────────────────────────────────────────────────

print("\nFitting TF-IDF vectorizer ...")
tfidf = TFIDFVectorizer(max_features=500, min_precision=0.65)
X     = tfidf.fit_transform(df['tokens'].tolist(), df['label_num'].tolist())
y     = df['label_num'].values

print(f"\nTop spam-signal words in vocabulary:")
print(f"  {tfidf.index_word_[:25]}")

# sklearn train_test_split with stratify to keep class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain : {len(X_train)}  (spam={y_train.sum()})")
print(f"Test  : {len(X_test)}   (spam={y_test.sum()})")

spam_hits = X_test[y_test == 1].sum(1)
ham_hits  = X_test[y_test == 0].sum(1)
print(f"\nVocab hits per email:")
print(f"  Spam — min:{int(spam_hits.min())}  median:{int(np.median(spam_hits))}  max:{int(spam_hits.max())}")
print(f"  Ham  — min:{int(ham_hits.min())}   p90:{int(np.percentile(ham_hits, 90))}   max:{int(ham_hits.max())}")
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
print(f"\nAfter oversampling — Ham: {int((y_bal == 0).sum())}  Spam: {int((y_bal == 1).sum())}")

# ─────────────────────────────────────────────────────────────
# STEP 5 — Hardcoded Decision Tree (CART, Gini impurity)
# ─────────────────────────────────────────────────────────────

class DecisionTree:
    """
    Hardcoded CART Decision Tree for binary count features.

    Split: threshold = 0.0
        Binary features are exactly 0 (word absent) or 1 (word present).
        col <= 0.0  → left  branch: word absent  → likely ham
        col >  0.0  → right branch: word present → likely spam
        Only one threshold needed per feature (binary data).

    Weighted Gini impurity upweights the minority class (spam) so the
    tree is penalised for missing spam, not just for being inaccurate.
    """

    def __init__(self, max_depth=20, min_samples_split=2,
                 max_features=None, class_weight='balanced'):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.class_weight      = class_weight
        self.tree_             = None
        self.cw_               = {}

    def _gini(self, y):
        n = len(y)
        if n == 0:
            return 0.0
        return 2.0 * sum(
            self.cw_.get(int(c), 1.0) * (y == c).sum() / n
            * (1.0 - (y == c).sum() / n)
            for c in self.cw_
        )

    def _best_split(self, X, y, feat_ids):
        best_gain   = -math.inf
        best_feat   = None
        pg          = self._gini(y)
        n           = len(y)

        for f in feat_ids:
            col = X[:, f]
            # Binary features: only valid threshold is 0.0
            lm = col <= 0.0
            rm = ~lm
            nl, nr = lm.sum(), rm.sum()
            if nl == 0 or nr == 0:
                continue
            gain = pg - (nl * self._gini(y[lm]) + nr * self._gini(y[rm])) / n
            if gain > best_gain:
                best_gain = gain
                best_feat = f

        return best_feat, 0.0, best_gain

    def _build(self, X, y, depth):
        n = len(y)
        if (depth >= self.max_depth
                or n < self.min_samples_split
                or len(np.unique(y)) == 1):
            return {'leaf': True,
                    'cls' : int(np.bincount(y, minlength=2).argmax())}

        d     = X.shape[1]
        k     = min(self.max_features or d, d)
        f_ids = np.random.choice(d, k, replace=False)

        feat, thresh, gain = self._best_split(X, y, f_ids)
        if feat is None or gain <= 1e-9:
            return {'leaf': True,
                    'cls' : int(np.bincount(y, minlength=2).argmax())}

        lm = X[:, feat] <= thresh
        return {
            'leaf'  : False,
            'feat'  : feat,
            'thresh': thresh,
            'left'  : self._build(X[ lm], y[ lm], depth + 1),
            'right' : self._build(X[~lm], y[~lm], depth + 1),
        }

    def fit(self, X, y):
        classes, counts = np.unique(y, return_counts=True)
        n = len(y)
        self.cw_ = ({int(c): n / (len(classes) * cnt)
                     for c, cnt in zip(classes, counts)}
                    if self.class_weight == 'balanced'
                    else {int(c): 1.0 for c in classes})
        self.tree_ = self._build(X, y, 0)
        return self

    def _walk(self, x, node):
        if node['leaf']:
            return node['cls']
        return (self._walk(x, node['left'])
                if x[node['feat']] <= node['thresh']
                else self._walk(x, node['right']))

    def predict(self, X):
        return np.array([self._walk(x, self.tree_) for x in X])

# ─────────────────────────────────────────────────────────────
# STEP 6 — Hardcoded Random Forest
# ─────────────────────────────────────────────────────────────

class RandomForest:
    """
    Hardcoded Random Forest: ensemble of Decision Trees.

    Training:
        Each tree is trained on a bootstrap sample (sampling with replacement).
        Each split considers only sqrt(n_features) randomly chosen features.

    Prediction (Majority Voting):
        Collect votes from all trees.
        spam_fraction = trees_voting_spam / total_trees
        Predict SPAM if spam_fraction >= vote_threshold, else HAM.
        vote_threshold=0.45 slightly below 0.5 to improve recall on
        borderline spam while keeping precision high.
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
        n, d = X.shape
        k = (max(1, int(math.sqrt(d)))   if self.max_features == 'sqrt'  else
             max(1, int(math.log2(d)))   if self.max_features == 'log2'  else d)

        print(f"\nTraining Random Forest ({self.n_trees} trees, "
              f"max_depth={self.max_depth}, "
              f"features_per_split={k}, "
              f"vote_threshold={self.vote_threshold}) ...")

        for i in range(self.n_trees):
            # Bootstrap sample (with replacement)
            boot_idx = np.random.choice(n, n, replace=True)
            tree = DecisionTree(
                max_depth         = self.max_depth,
                min_samples_split = self.min_samples_split,
                max_features      = k,
                class_weight      = self.class_weight,
            )
            tree.fit(X[boot_idx], y[boot_idx])
            self.trees_.append(tree)
            print(f"  Tree {i+1:>2}/{self.n_trees} done", end='\r')

        print(f"\nAll {self.n_trees} trees trained.")
        return self

    def predict(self, X):
        """
        Majority Voting:
            votes shape : (n_trees, n_samples)
            spam_frac   : fraction of trees that voted SPAM per sample
        """
        votes     = np.vstack([t.predict(X) for t in self.trees_])
        spam_frac = votes.mean(axis=0)
        return (spam_frac >= self.vote_threshold).astype(int)

# ─────────────────────────────────────────────────────────────
# STEP 7 — Train
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
# STEP 8 — Evaluation Metrics (sklearn)
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
# STEP 9 — Predict New Emails
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
