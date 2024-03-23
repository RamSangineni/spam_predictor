import streamlit as st
import pickle
import re
import nltk
import joblib
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from io import BytesIO

# Function to fetch model from GitHub
def fetch_model_from_github(model_url):
    response = requests.get(model_url)
    model_bytes = BytesIO(response.content)
    model = joblib.load(model_bytes)
    return model

# Function to fetch TF-IDF vectorizer from GitHub
def fetch_vectorizer_from_github(vectorizer_url):
    response = requests.get(vectorizer_url)
    vectorizer_bytes = BytesIO(response.content)
    vectorizer = joblib.load(vectorizer_bytes)
    return vectorizer

# Load the trained model from GitHub
model_url = "https://github.com/RamSangineni/spam_predictor/raw/main/Bernouli_Model.pkl"
model = fetch_model_from_github(model_url)

# Load TF-IDF vectorizer from GitHub
vectorizer_url = "https://github.com/RamSangineni/spam_predictor/raw/main/TFIDF_vectorizer.pkl"
tfidf = fetch_vectorizer_from_github(vectorizer_url)

# Function to remove special characters and punctuation
def cleaning_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    white_space = re.sub(r'\s+', ' ', cleaned_text)
    lowered_text = white_space.lower()
    return lowered_text

# Function for stemming to cut the words to their root form
def perform_stemming(words):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in words.split()]
    return ' '.join(stemmed_tokens)

# removing stopwords
def remove_stopwords(words):
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words.split() if word not in stop_words]
    return ' '.join(filtered_words)

# calling all the functions into this main function
def cleaning_preprocessing(text):
    cleaned = cleaning_text(text)
    stemmed = perform_stemming(cleaned)
    without_stopwords = remove_stopwords(stemmed)
    return without_stopwords

# Function to classify emails
def prediction(MSG: str):
    vectorized_email = tfidf.transform([MSG]).toarray()
    prediction = model.predict(vectorized_email)
    probabilities = model.predict_proba(vectorized_email)
    spam_probability = probabilities[0][1] * 100
    ham_probability = probabilities[0][0] * 100
    return prediction, spam_probability, ham_probability

# Page title and description
st.title('Message Classification App')
st.write('This app classifies Messages into spam or not spam.')

message = st.text_input('Enter your Message here')
message = cleaning_preprocessing(message)

if st.button("Predict"):
    result, spam_prob, ham_prob = prediction(message)
    if result == 1:
        st.write('This message is a Spam with a probability of {:.2f}%'.format(spam_prob))
    else:
        st.write('This message is not Spam with a probability of {:.2f}%'.format(ham_prob))
