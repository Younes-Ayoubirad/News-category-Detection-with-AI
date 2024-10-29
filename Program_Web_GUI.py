import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
from hazm import word_tokenize
from hazm.stemmer import Stemmer
import pickle


LE_saved_file = open('saved_models/LabelEncoder_save.pkl', 'rb')
LabelEncoder_model = pickle.load(LE_saved_file)
LE_saved_file.close()

VC_saved_file = open('saved_models/tfidf_vectorizer_save.pkl', 'rb')
vectorizer_model = pickle.load(VC_saved_file)
VC_saved_file.close()

ml_saved_file = open('saved_models/model_save.pkl', 'rb')
svm_model = pickle.load(ml_saved_file)
ml_saved_file.close()

with open('saved_models/All_stopwords_save.pkl', 'rb') as f:
    All_stopwords = pickle.load(f)

lemmatizer = WordNetLemmatizer()
stemmer = Stemmer()

def preprocess_text(text, stemmer, stopwords):
    tokenized_text = word_tokenize(text)
    filtered_text = [w for w in tokenized_text if w not in stopwords]
    stemmed_text = [stemmer.stem(w) for w in filtered_text]
    return [' '.join(stemmed_text)]

def predict_class(news):
    x = preprocess_text(news, stemmer, All_stopwords)
    x_v = vectorizer_model.transform(x)
    p = svm_model.predict(x_v)
    label = LabelEncoder_model.inverse_transform(p)
    return label[0]


st.title('News category Detection with AI')

text = st.text_area('Enter your news text: ')

btn = st.button('Detect Category')

if btn:
    out_text = predict_class(text)
    st.success("news category is :  " + out_text)