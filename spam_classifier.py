import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only the useful columns (v1 = label, v2 = message)
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# 2. Convert labels (ham → 0, spam → 1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# 4. Convert text into numerical features
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# 5. Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# 6. Test the model
y_pred = model.predict(X_test_counts)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 7. Function to classify new emails
def classify_email(text):
    counts = vectorizer.transform([text])
    prediction = model.predict(counts)
    return "Spam" if prediction[0] == 1 else "Ham"

# Example usage
print("Sample Email Prediction:", classify_email("Congratulations! You won a free lottery ticket, click here!"))
print("Sample Email Prediction:", classify_email("Hey Shree, let's meet tomorrow at 5 PM"))

# 8. Allow user input from terminal
user_input = input("Type an email message: ")
print("Your Email Prediction:", classify_email(user_input))

# 9. (Optional) Visualize top spam words
spam_messages = data[data['label'] == 1]['message']
spam_counts = vectorizer.transform(spam_messages)
word_freq = np.sum(spam_counts, axis=0)
words = vectorizer.get_feature_names_out()
freq_df = pd.DataFrame(word_freq.T, index=words, columns=['count'])
top_words = freq_df.sort_values('count', ascending=False).head(20)

sns.barplot(x=top_words['count'], y=top_words.index)
plt.title("Top Spam Words")
plt.show()
