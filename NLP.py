import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
data = {
    'review': [
        "I love this product, it works great!",
        "Absolutely terrible. Waste of money.",
        "Decent quality, could be better.",
        "Amazing experience, very satisfied!",
        "Worst purchase ever. Very bad.",
        "I'm happy with the quality.",
        "Not what I expected. Disappointed.",
        "Superb! Exceeded my expectations.",
        "Poor quality and bad customer service.",
        "Will buy again, very good!"
    ],
    'sentiment': [
        'positive',
        'negative',
        'neutral',
        'positive',
        'negative',
        'positive',
        'negative',
        'positive',
        'negative',
        'positive'
    ]
}
df = pd.DataFrame(data)
df['sentiment'] = df['sentiment'].replace('neutral', 'positive')
X = df['review']
y = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
