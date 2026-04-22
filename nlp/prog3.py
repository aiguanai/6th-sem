# Program 3: Sentiment Analysis using Manual TF-IDF + Logistic Regression
# Applied to Musical Instruments Reviews Dataset
# (No inbuilt TF-IDF library used)

import pandas as pd
import numpy as np
import math
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------
# STEP 1: Load and Prepare Dataset
# ---------------------------------------------------

print("Loading dataset...")
df = pd.read_csv('Musical_instruments_reviews_4.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# Create sentiment labels from ratings
def rating_to_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating <= 2:
        return "Negative"
    else:
        return "Neutral"

df['Sentiment'] = df['overall'].apply(rating_to_sentiment)

print("Sentiment Distribution:")
print(df['Sentiment'].value_counts())
print()

# ---------------------------------------------------
# STEP 2: Text Cleaning
# ---------------------------------------------------

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

print("Cleaning text...")
df["Cleaned"] = df["reviewText"].apply(clean_text)

df = df[df["Cleaned"].str.len() > 0]
print(f"Reviews after cleaning: {len(df)}\n")

# ---------------------------------------------------
# STEP 3: Train Test Split
# ---------------------------------------------------

print("Splitting data into train and test sets (70-30 split)...")
X_train, X_test, y_train, y_test = train_test_split(
    df["Cleaned"], 
    df["Sentiment"], 
    test_size=0.3, 
    random_state=42,
    stratify=df["Sentiment"]
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Training set sentiment distribution:\n{y_train.value_counts()}\n")

# ---------------------------------------------------
# STEP 4: Build Vocabulary (with filtering for performance)
# ---------------------------------------------------

print("Building vocabulary...")

word_doc_freq = Counter()

for sentence in X_train:
    words = set(sentence.split())
    word_doc_freq.update(words)

min_doc_freq = 2
max_doc_freq = int(0.8 * len(X_train))

vocab = sorted([word for word, freq in word_doc_freq.items() 
                if min_doc_freq <= freq <= max_doc_freq])

print(f"Vocabulary size: {len(vocab)} (filtered from {len(word_doc_freq)} unique words)\n")

# ---------------------------------------------------
# STEP 5: Compute IDF Manually (Optimized)
# ---------------------------------------------------

print("Computing IDF (Inverse Document Frequency)...")
N = len(X_train)
idf = {}

X_train_list = X_train.tolist()

for idx, word in enumerate(vocab):
    doc_count = 0
    for sentence in X_train_list:
        if word in sentence.split():
            doc_count += 1
    
    idf[word] = math.log(N / (1 + doc_count))
    
    if (idx + 1) % 1000 == 0:
        print(f"  Processed {idx + 1}/{len(vocab)} words...")

print(f"IDF computation complete.\n")

# ---------------------------------------------------
# STEP 6: Compute TF-IDF Manually
# ---------------------------------------------------

def tfidf_vector(sentence):
    words = sentence.split()
    total_words = len(words)
    
    if total_words == 0:
        return [0] * len(vocab)
    
    word_freq = Counter(words)
    vector = []

    for word in vocab:
        tf = word_freq[word] / total_words
        tfidf = tf * idf[word]
        vector.append(tfidf)

    return vector

print("Computing TF-IDF vectors for training data...")
X_train_list = X_train.tolist()
X_train_tfidf = []

for idx, text in enumerate(X_train_list):
    vector = tfidf_vector(text)
    X_train_tfidf.append(vector)
    
    if (idx + 1) % 2000 == 0:
        print(f"  Processed {idx + 1}/{len(X_train_list)} training documents...")

X_train_tfidf = np.array(X_train_tfidf)
print(f"Training TF-IDF matrix shape: {X_train_tfidf.shape}")

print("Computing TF-IDF vectors for test data...")
X_test_list = X_test.tolist()
X_test_tfidf = []

for idx, text in enumerate(X_test_list):
    vector = tfidf_vector(text)
    X_test_tfidf.append(vector)
    
    if (idx + 1) % 1000 == 0:
        print(f"  Processed {idx + 1}/{len(X_test_list)} test documents...")

X_test_tfidf = np.array(X_test_tfidf)
print(f"Test TF-IDF matrix shape: {X_test_tfidf.shape}\n")

# ---------------------------------------------------
# STEP 7: Logistic Regression Model
# ---------------------------------------------------

print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42)
model.fit(X_train_tfidf, y_train)
print("Model training complete.\n")

# ---------------------------------------------------
# STEP 8: Prediction
# ---------------------------------------------------

print("Making predictions on test set...")
y_pred = model.predict(X_test_tfidf)

# ---------------------------------------------------
# STEP 9: Evaluation
# ---------------------------------------------------

print("=" * 80)
print("MODEL EVALUATION RESULTS")
print("=" * 80)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n" + "-" * 80)
print("Classification Report:")
print("-" * 80)
print(classification_report(y_test, y_pred))

print("-" * 80)
print("Confusion Matrix:")
print("-" * 80)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print()

unique_labels = sorted(y_test.unique())
print("Confusion Matrix Interpretation:")
print("Rows = Actual, Columns = Predicted")
print(f"Classes: {unique_labels}\n")

# ---------------------------------------------------
# STEP 10: Test with New Reviews
# ---------------------------------------------------

print("=" * 80)
print("TESTING WITH NEW REVIEWS")
print("=" * 80)

new_reviews = [
    "The guitar cables are amazing and very durable. Best quality for the price!",
    "Terrible product. Broke after just two weeks. Complete waste of money.",
    "It's okay. Does what it's supposed to do but nothing special.",
    "Excellent sound quality and fantastic build. Highly recommend!",
    "Poor quality, not worth buying. Disappointed with this purchase.",
    "Average product. Works fine for the price point.",
    "Absolutely love this! Great quality and fast shipping.",
    "Not satisfied. Quality could be much better.",
    "It's decent. Not the best but acceptable for casual use.",
    "Outstanding performance! Better than expected. Five stars!"
]

print(f"\nCleaning and processing {len(new_reviews)} new reviews...\n")

new_reviews_clean = [clean_text(text) for text in new_reviews]
new_vectors = np.array([tfidf_vector(text) for text in new_reviews_clean])

predictions = model.predict(new_vectors)
probabilities = model.predict_proba(new_vectors)

print("-" * 80)
print("New Review Predictions:")
print("-" * 80)

for i, (review, sentiment, probs) in enumerate(zip(new_reviews, predictions, probabilities)):
    print(f"\nReview {i+1}:")
    print(f"Text: {review}")
    print(f"Predicted Sentiment: {sentiment}")
    
    print(f"Confidence Scores:")
    for label, prob in zip(model.classes_, probs):
        print(f"  {label}: {prob:.4f} ({prob*100:.2f}%)")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

# ---------------------------------------------------
# STEP 11: Additional Analysis
# ---------------------------------------------------

print("\n" + "=" * 80)
print("ADDITIONAL STATISTICS")
print("=" * 80)

print(f"\nTest Set Performance by Sentiment Class:")
for label in unique_labels:
    mask = y_test == label
    class_accuracy = accuracy_score(y_test[mask], y_pred[mask])
    count = mask.sum()
    print(f"  {label}: Accuracy = {class_accuracy:.4f}, Count = {count}")

print(f"\nTop 10 Most Important Words (by average IDF):")
top_words = sorted(idf.items(), key=lambda x: x[1], reverse=True)[:10]
for word, idf_val in top_words:
    print(f"  {word}: {idf_val:.4f}")

print("\n" + "=" * 80)
