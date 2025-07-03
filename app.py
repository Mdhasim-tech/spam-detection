import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📧 Spam Detection App")
st.write("Enter a message to classify it as **Spam** or **Ham (Not Spam)**")

# Text input
message = st.text_area("Enter your message here:")

# Prediction
if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message first!")
    else:
        msg_transformed = vectorizer.transform([message])
        prediction = model.predict(msg_transformed)
        if prediction[0] == 1:
            st.error("🔴 This is SPAM!")
        else:
            st.success("🟢 This is NOT SPAM.")
