import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("dataset/spam.csv", encoding="latin-1")

# Keep only required columns
df = df[['v1', 'v2']]

# Rename columns
df.columns = ['label', 'message']

# Convert labels into numbers
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

# Features and Labels
X = df['message']
y = df['label']

# Convert text into numerical vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save trained model
with open("models/spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save vectorizer
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Predict on test data
prediction = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, prediction)

print("=" * 50)
print("SPAM EMAIL CLASSIFIER")
print("=" * 50)

print("\nModel Accuracy :", round(accuracy * 100, 2), "%")

# Confusion Matrix
print("\nConfusion Matrix")
print(confusion_matrix(y_test, prediction))

# Classification Report
print("\nClassification Report")
print(classification_report(y_test, prediction))

# User Prediction
print("=" * 50)

while True:

    user_message = input("\nEnter your message (or type exit): ")

    if user_message.lower() == "exit":
        print("Program Closed.")
        break

    user_vector = vectorizer.transform([user_message])

    result = model.predict(user_vector)

    if result[0] == 1:
        print("\nPrediction : SPAM")
    else:
        print("\nPrediction : HAM")