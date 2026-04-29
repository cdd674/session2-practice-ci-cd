# Task 3: Logistic Regression with GloVe Embeddings

## Objective

We convert IMDB reviews into vectors using pretrained GloVe embeddings.

Then we use Logistic Regression from sklearn for sentiment classification:

* 1 = Positive
* 0 = Negative

---

# Step 1: Import Libraries

```python
import numpy as np
import pandas as pd
import json
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

---

# Step 2: Load Dataset

```python
df = pd.read_csv("./datasets/imdb_top_500.csv")

with open("./datasets/tiny_glove.json", "r") as f:
    glove = json.load(f)
```

```python
print("Dataset size:", len(df))
print("Columns:", df.columns.tolist())

print("\nFirst review preview:")
print(df["text"].iloc[0][:300])

print("\nFirst label:", df["label"].iloc[0])
print("First rating:", df["rating"].iloc[0])

print("\nVocabulary size:", len(glove))
```

---

# Step 3: Clean and Tokenize Text

```python
def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()
```

```python
sample_tokens = tokenize(df["text"].iloc[0])

print("First 20 tokens:")
print(sample_tokens[:20])

print("Total token count:", len(sample_tokens))
```

---

# Step 4: Convert One Review into One Vector

```python
def get_embedding(text, glove, dim=50):
    tokens = tokenize(text)

    vectors = [
        np.array(glove[word])
        for word in tokens
        if word in glove
    ]

    if len(vectors) == 0:
        return np.zeros(dim)

    return np.mean(vectors, axis=0)
```

```python
sample_vector = get_embedding(df["text"].iloc[0], glove)

print("Embedding shape:", sample_vector.shape)

print("First 10 embedding values:")
print(sample_vector[:10])
```

---

# Step 5: Build Features, Labels, and Original Text Together

This is important.

We keep the original review text so later predictions match the correct review.

```python
X = np.array([get_embedding(text, glove) for text in df["text"]])

y = df["label"].values

texts = df["text"].values
```

```python
print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)
print("Texts shape:", texts.shape)

print("Positive ratio:", np.mean(y))
```

---

# Step 6: Split Train and Test Data Correctly

```python
X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
    X,
    y,
    texts,
    test_size=0.2,
    random_state=42
)
```

```python
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nFirst test review preview:")
print(text_test[0][:300])
```

---

# Step 7: Standardize Features

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
```

```python
print("Feature scaling complete.")

print("First standardized training vector:")
print(X_train[0][:10])
```

---

# Step 8: Train Logistic Regression

```python
model = LogisticRegression()

model.fit(X_train, y_train)
```

```python
print("Model training complete.")
```

---

# Step 9: Make Predictions

```python
train_pred = model.predict(X_train)

test_pred = model.predict(X_test)
```

```python
print("First 30 test predictions:")
print(test_pred[:30])

print("\nFirst 30 true labels:")
print(y_test[:30])
```

---

# Step 10: Evaluate Accuracy

```python
train_acc = accuracy_score(y_train, train_pred)

test_acc = accuracy_score(y_test, test_pred)
```

```python
print("Train Accuracy:", train_acc)

print("Test Accuracy:", test_acc)
```

---

# Step 11: Compare Real Test Reviews with Predictions

Now each prediction matches the correct original review.

```python
for i in range(85, 90):
    print(f"\nReview {i+1}")
    print("-" * 60)

    print(text_test[i][:400])

    print("\nTrue Label:", y_test[i])

    print("Predicted Label:", test_pred[i])

    if y_test[i] == test_pred[i]:
        print("Result: Correct")
    else:
        print("Result: Incorrect")
```

---

# Step 12: Predict Sentiment for New Reviews

```python
sample_reviews = [
    "This movie was fantastic with brilliant acting",
    "I hated this movie it was boring and terrible",
    "The film was okay not great but not bad"
]
```

```python
sample_X = np.array([
    get_embedding(text, glove)
    for text in sample_reviews
])

sample_X = scaler.transform(sample_X)

sample_preds = model.predict(sample_X)
```

```python
for review, pred in zip(sample_reviews, sample_preds):
    print("\nReview:")
    print(review)

    print(
        "Predicted Sentiment:",
        "Positive" if pred == 1 else "Negative"
    )
```
