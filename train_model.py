import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# Load and combine datasets
df1 = pd.read_csv('SMSSpamCollection.tsv', sep='\t', names=['label', 'message'])
df2 = pd.read_csv('SinhalaSpamCollection.tsv', sep='\t', names=['label', 'message'])
df = pd.concat([df1, df2], ignore_index=True)

# Clean and shuffle
df.dropna(inplace=True)
df = df.sample(frac=1).reset_index(drop=True)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Optional: Evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained and saved using SMSSpamCollection.tsv and SinhalaSpamCollection.tsv")
