# Sentiment Analysis using
# Manual Bag of Words + Hardcoded Naive Bayes
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

import pandas as pd
import math
import re
import nltk
import random

from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report, confusion_matrix


# ---------------------------------------------------
# STEP 1 : LOAD DATASET
# ---------------------------------------------------

df = pd.read_csv("/content/Musical_instruments_reviews 4.csv")

# ---------------------------------------------------
# STEP 2 : RATING TO SENTIMENT
# 0 = Negative  |  1 = Neutral  |  2 = Positive
# ---------------------------------------------------

def rating_to_sentiment(rating):
    if rating >= 4:
        return 2
    elif rating == 3:
        return 1
    else:
        return 0

df["Sentiment"] = df["overall"].apply(rating_to_sentiment)

# ---------------------------------------------------
# STEP 3 : TEXT CLEANING
# ---------------------------------------------------

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)                          # changed from text.split()
    words = [w for w in words if w not in stop_words]    
    return " ".join(words)

df["Cleaned"] = df["reviewText"].apply(clean_text)
df = df[df["Cleaned"].str.strip() != ""]

# ---------------------------------------------------
# STEP 4 : TRAIN TEST SPLIT (stratified)
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["Cleaned"],
    df["Sentiment"],
    test_size=0.3,
    random_state=42,
    stratify=df["Sentiment"]        # keeps class ratios equal in both splits
)

# ---------------------------------------------------
# STEP 5 : OVERSAMPLE MINORITY CLASSES IN TRAINING SET
# This is the core fix — we repeat Negative and Neutral
# samples until all three classes have equal counts.
# ---------------------------------------------------

train_df = pd.DataFrame({"text": X_train, "label": y_train})

max_count = train_df["label"].value_counts().max()

balanced_parts = []
for cls in train_df["label"].unique():
    cls_df = train_df[train_df["label"] == cls]
    # repeat rows until we reach max_count
    oversampled = cls_df.sample(n=max_count, replace=True, random_state=42)
    balanced_parts.append(oversampled)

train_balanced = pd.concat(balanced_parts).sample(frac=1, random_state=42)

X_train_bal = train_balanced["text"].tolist()
y_train_bal = train_balanced["label"].tolist()

print("Balanced class counts:")
print(pd.Series(y_train_bal).value_counts())

# ---------------------------------------------------
# STEP 6 : BUILD VOCABULARY (from balanced training set)
# ---------------------------------------------------

vocab = set()
for sentence in X_train_bal:
    vocab.update(sentence.split())
vocab = sorted(list(vocab))

# ---------------------------------------------------
# STEP 7 : NAIVE BAYES CLASSIFIER
# ---------------------------------------------------

class NaiveBayes:

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_word_count = defaultdict(lambda: defaultdict(int))
        self.class_count      = defaultdict(int)
        self.total_words      = defaultdict(int)
        self.total_docs       = 0
        self.vocab_size       = 0
        self.classes          = set()

    def fit(self, texts, labels):
        self.total_docs  = len(texts)
        self.vocab_size  = len(vocab)

        for text, label in zip(texts, labels):
            self.classes.add(label)
            self.class_count[label] += 1
            for word in text.split():
                self.class_word_count[label][word] += 1
                self.total_words[label] += 1

    def predict_one(self, text):
        words      = text.split()
        best_class = None
        best_score = float("-inf")

        for cls in self.classes:
            prior      = math.log(self.class_count[cls] / self.total_docs)
            likelihood = 0

            for word in words:
                count  = self.class_word_count[cls][word]
                prob   = (count + self.alpha) / (
                             self.total_words[cls] + self.alpha * self.vocab_size
                         )
                likelihood += math.log(prob)

            score = prior + likelihood
            if score > best_score:
                best_score = score
                best_class = cls

        return best_class

    def predict(self, texts):
        return [self.predict_one(t) for t in texts]

# ---------------------------------------------------
# STEP 8 : TRAIN
# ---------------------------------------------------

model = NaiveBayes(alpha=1.0)
model.fit(X_train_bal, y_train_bal)

# ---------------------------------------------------
# STEP 9 : EVALUATE  (on original, unbalanced test set)
# ---------------------------------------------------

y_pred = model.predict(X_test.tolist())

print("\nAccuracy  :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred, average="weighted"))
print("Recall    :", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score  :", f1_score(y_test, y_pred, average="weighted"))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred,
                             target_names=["Negative", "Neutral", "Positive"]))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------
# STEP 10 : INFERENCE
# ---------------------------------------------------

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

new_reviews = [
    "This guitar sounds amazing, the tone is perfect and build quality is excellent",
    "Terrible product, strings broke after one day and sound is awful",
    "The keyboard is decent, nothing special but gets the job done"
]

cleaned_reviews = [clean_text(r) for r in new_reviews]
predictions     = model.predict(cleaned_reviews)

print("\nNew Predictions:\n")
for review, pred in zip(new_reviews, predictions):
    print("Review   :", review)
    print("Sentiment:", label_map[pred])
    print()
