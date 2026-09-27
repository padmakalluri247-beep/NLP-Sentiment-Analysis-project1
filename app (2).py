
import streamlit as st
import joblib
import re

# Load saved model and TF-IDF vectorizer
model = joblib.load("svm_sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


# Same preprocessing used during model training
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Streamlit page
st.set_page_config(
    page_title="NLP Sentiment Analysis",
    page_icon="😊"
)

st.title("NLP Sentiment Analysis")
st.write("Analyze the sentiment of a customer review.")


# User input
review = st.text_area(
    "Enter Customer Review:",
    placeholder="Type your product review here..."
)


# Prediction button
if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a customer review.")

    else:
        # Clean the review
        cleaned_review = clean_text(review)

        # Convert text into TF-IDF features
        review_tfidf = tfidf.transform([cleaned_review])

        # Predict sentiment
        prediction = model.predict(review_tfidf)[0]

        # Display result
        st.subheader("Prediction")
        st.success(f"Sentiment: {prediction}")
