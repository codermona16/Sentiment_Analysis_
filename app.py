import streamlit as st
import pickle
import gdown
import os


MODEL_FILE_ID = "1ssbYGj7afMLptc9PKGNKCSnmY7JZXKPn"
MODEL_FILE = "best_model.pkl"

def download_if_not_exists(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, output, quiet=False)

def load_pickle(filepath):
    with open(filepath, "rb") as file:
        return pickle.load(file)

download_if_not_exists(MODEL_FILE_ID, MODEL_FILE)


model = load_pickle(MODEL_FILE)

vectorizer = load_pickle("tfidf_vectorizer.pkl")
le = load_pickle("label_encoder.pkl")





def predict_sentiment(text):
    text_vect = vectorizer.transform([text])  # Transform user input
    pred_encoded = model.predict(text_vect)   # Predict encoded label
    pred_label = le.inverse_transform(pred_encoded)  # Convert back to text label
    return pred_label[0]


st.title("📊 Sentiment Analysis App")
st.write("Enter a review or any text below to analyze its sentiment.")

user_input = st.text_area("📝 Enter your text here:")

if st.button("Predict Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.success(f"✅ **Predicted Sentiment:** {sentiment}")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
