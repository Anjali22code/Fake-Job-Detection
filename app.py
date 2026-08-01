from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Load Model
model = pickle.load(open("fake_job_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))


# Text Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    job_description = request.form["job_description"]

    job_description = clean_text(job_description)

    job_vector = vectorizer.transform([job_description])

    probability = model.predict_proba(job_vector)[0][1]

    if probability > 0.3:
        result = "⚠️ Fake Job Posting Detected"
        status = "fake"
    else:
        result = "✅ Legitimate Job Posting"
        status = "legit"

    return render_template(
        "index.html",
        prediction=result,
        probability=round(probability * 100, 2),
        status=status
    )


if __name__ == "__main__":
    app.run(debug=True)


