import streamlit as st
import pickle


# Load model and vectorizer

model = pickle.load(open("fake_job_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))


# Page title

st.title("Fake Job Posting Detection")

st.write(
    "This system uses Machine Learning to detect whether a job posting is fake or legitimate."
)


# Input box

job_description = st.text_area(
    "Enter Job Description"
)


# Prediction button

if st.button("Predict"):

    if job_description.strip() != "":

        job_vector = vectorizer.transform([job_description])

        prediction = model.predict(job_vector)[0]

        probability = model.predict_proba(job_vector)[0][1]


        if prediction == 1:

            st.error("⚠️ Fake Job Posting Detected")

        else:

            st.success("✅ Legitimate Job Posting")


        st.write(
            f"Fake Probability: {probability*100:.2f}%"
        )

    else:

        st.warning("Please enter job description")