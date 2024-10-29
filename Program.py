import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import svm
import nltk
from nltk.corpus import stopwords
import hazm
import os

# nltk.download('stopwords')


# Function to load stopwords
def load_stopwords(file_path, extra_stopwords=None):
    with open(file_path, encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
    if extra_stopwords:
        stopwords.extend(extra_stopwords)
    return set(stopwords)


# Text preprocessing function
def preprocess_text(text, stemmer, stopwords):
    tokenized_text = hazm.word_tokenize(text)
    filtered_text = [w for w in tokenized_text if w not in stopwords]
    stemmed_text = [stemmer.stem(w) for w in filtered_text]
    return ' '.join(stemmed_text)


# Main data processing function
def prepare_dataset(data, stopwords, stemmer):
    dataset = pd.DataFrame(columns=['title_body', 'category'])
    for index, row in data.iterrows():
        title_body = f"{row['title']} {row['description']}"
        processed_text = preprocess_text(title_body, stemmer, stopwords)
        dataset.loc[index] = {
            'title_body': processed_text,
            'category': row['tags'].strip()
        }
    return dataset


# Load and process data
data = pd.read_csv('data/tasnim_news.csv', encoding='utf-8')

# Load stopwords
All_stopwords = load_stopwords('data/stopwords.txt', nltk.corpus.stopwords.words('english'))

with open("saved_models/All_stopwords_save.pkl", 'wb') as f:
    pickle.dump(All_stopwords, f)

# Initialize the Hazm stemmer
stemmer = hazm.Stemmer()

# Prepare the dataset
dataset = prepare_dataset(data, All_stopwords, stemmer)

# Vectorize the text data using TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit the dataset text
tfidf_vectorizer.fit(dataset['title_body'])

# Save the fitted vectorizer for future use
with open("saved_models/tfidf_vectorizer_save.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

x = tfidf_vectorizer.transform(dataset['title_body'])


# Encode the categories using LabelEncoder
le = LabelEncoder()
y = le.fit_transform(dataset['category'])

# Save the fitted LabelEncoder for future use
with open("saved_models/LabelEncoder_save.pkl", "wb") as f:
    pickle.dump(le, f)

# Split the dataset into training and testing sets with a random state for reproducibility
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the SVM model
model = svm.SVC(kernel='linear')  # Consider setting a kernel for better performance
model.fit(x_train, y_train)

# Save the trained model
with open("saved_models/model_save.pkl", "wb") as f:
    pickle.dump(model, f)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Print classification report
print(classification_report(y_test, y_pred, zero_division=1))

