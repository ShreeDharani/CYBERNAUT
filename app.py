from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_counts, y_train)

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    css_class = ""
    spam_conf = 0
    ham_conf = 0
    if request.method == "POST":
        user_input = request.form["email_text"]
        counts = vectorizer.transform([user_input])
        prediction = model.predict(counts)
        probability = model.predict_proba(counts)[0]

        result = "Spam" if prediction[0] == 1 else "Ham"
        css_class = "spam" if result == "Spam" else "ham"

        # Confidence values
        spam_conf = round(probability[1] * 100)
        ham_conf = round(probability[0] * 100)

    return render_template("index.html",
                           result=result,
                           css_class=css_class,
                           spam_conf=spam_conf,
                           ham_conf=ham_conf)

if __name__ == "__main__":
    app.run(debug=True)
