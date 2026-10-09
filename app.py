import os
from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Example: simple spam classifier setup
vectorizer = CountVectorizer()
model = MultinomialNB()

# Dummy training data (replace with your dataset)
texts = ["Win money now", "Hello friend", "Claim your prize", "Meeting tomorrow"]
labels = [1, 0, 1, 0]  # 1 = spam, 0 = not spam

X = vectorizer.fit_transform(texts)
model.fit(X, labels)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    data = [message]
    vect = vectorizer.transform(data).toarray()
    prediction = model.predict(vect)
    result = "Spam" if prediction[0] == 1 else "Not Spam"
    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render provides PORT
    app.run(host="0.0.0.0", port=port)
