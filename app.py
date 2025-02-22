import streamlit as st
import pickle
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem.porter import PorterStemmer

# Load the pre-trained model and vectorizer
def load_models():
    try:
        vectorizer = pickle.load(open("vectorizer1.pkl", "rb"))
        model = pickle.load(open("model1.pkl", "rb"))
        return vectorizer, model
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        return None, None

vectorizer, model = load_models()

# Initialize Porter Stemmer
ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    words = [word for word in words if word not in ENGLISH_STOP_WORDS and word not in string.punctuation]
    words = [ps.stem(word) for word in words]
    return " ".join(words)

# Streamlit UI Styling
def set_custom_style():
    st.markdown(
        """
        <style>
        .stApp { 
            background: linear-gradient(to right, #ff7eb3, #ff758c);
            font-family: 'Arial', sans-serif;
            text-align: center;
        }
        .stButton>button {
            background: linear-gradient(135deg, #ff416c, #ff4b2b);
            color: white;
            border-radius: 30px;
            padding: 14px 28px;
            font-size: 20px;
            font-weight: bold;
            border: none;
            transition: 0.3s ease;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.25);
        }
        .stTextInput>div>div>input {
            border-radius: 15px;
            padding: 12px;
            border: 2px solid #ff4b2b;
            font-size: 18px;
            text-align: center;
        }
        .header-title {
            font-size: 40px;
            font-weight: bold;
            color: white;
            text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.2);
        }
        .stMarkdown p {
            font-size: 18px;
            color: #f5f5f5;
        }
        .result-box {
            background: rgba(255, 255, 255, 0.9);
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# App Title and Description
def display_header():
    st.markdown("""<h1 class='header-title'>📩 SMS Spam Detector</h1>""", unsafe_allow_html=True)
    st.markdown("""<p>🔍 Enter an SMS message to check if it's spam or not.</p>""", unsafe_allow_html=True)

# Input Section
def get_user_input():
    return st.text_input("Enter the SMS", placeholder="Type your SMS here...")

# Prediction Logic
def predict_spam(input_sms):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter an SMS to predict.")
        return
    
    with st.spinner("Analyzing..."):
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
        
        transformed_sms = transform_text(input_sms)
        vector_input = vectorizer.transform([transformed_sms])
        result = model.predict(vector_input)[0]
    
    st.markdown("---")
    if result == 1:
        st.markdown("""
            <div class='result-box' style='background-color: #ffebee; color: red;'>
                🚨 <b>This is a Spam SMS!</b><br>⚠️ Be cautious! This message is likely spam.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class='result-box' style='background-color: #e8f5e9; color: green;'>
                ✅ <b>This is Not Spam!</b><br>👍 This message looks safe.
            </div>
            """, unsafe_allow_html=True)

# Main Function
def main():
    set_custom_style()
    display_header()
    input_sms = get_user_input()
    if st.button('🚀 Predict'):
        predict_spam(input_sms)

if __name__ == "__main__":
    main()

