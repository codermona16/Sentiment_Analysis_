import streamlit as st
import pickle

def load_pickle(filepath):
    with open(filepath, "rb") as file:
        return pickle.load(file)

model = load_pickle("C:/Users/ASUS/OneDrive/Desktop/Sentiment_Analysis_/best_model.pkl")
vectorizer = load_pickle("C:/Users/ASUS/OneDrive/Desktop/Sentiment_Analysis_/tfidf_vectorizer.pkl")
le = load_pickle("C:/Users/ASUS/OneDrive/Desktop/Sentiment_Analysis_/label_encoder.pkl")



# ✅ Function to predict sentiment
def predict_sentiment(text):
    text_vect = vectorizer.transform([text])  # Transform user input
    pred_encoded = model.predict(text_vect)   # Predict encoded label
    pred_label = le.inverse_transform(pred_encoded)  # Convert back to text label
    return pred_label[0]

# ✅ Streamlit UI
st.title("📊 Sentiment Analysis App")
st.write("Enter a review or any text below to analyze its sentiment.")

user_input = st.text_area("📝 Enter your text here:")

if st.button("Predict Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.success(f"✅ **Predicted Sentiment:** {sentiment}")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
