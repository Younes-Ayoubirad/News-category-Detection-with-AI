import nltk
from nltk.stem import WordNetLemmatizer
from hazm import word_tokenize
from hazm.stemmer import Stemmer
import pickle
import os
import json
import random
from tkinter import *


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

def on_key_release(event):
    ctrl = (event.state & 0x4) != 0

    if event.keycode == 88 and ctrl and event.keysym.lower() == 'x':
        event.widget.event_generate('<<Cut>>')

    if event.keycode == 86 and ctrl and event.keysym.lower() == 'v':
        event.widget.event_generate('<<Paste>>')

    if event.keycode == 67 and ctrl and event.keysym.lower() == 'c':
        event.widget.event_generate('<<Copy>>')

def send():
    msg = InputBox.get("1.0", 'end-1c').strip()
    InputBox.delete('0.0', END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        res = predict_class(msg)
        ChatLog.insert(END, 'News Category is : ' + res + '\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


def predict_class(news):
    x = preprocess_text(news, stemmer, All_stopwords)
    x_v = vectorizer_model.transform(x)
    p = svm_model.predict(x_v)
    label = LabelEncoder_model.inverse_transform(p)
    return label[0]


base = Tk()
base.title('News Category')
base.geometry('400x500')
base.resizable(width=False, height=False)

ChatLog = Text(base, bd=1, bg='gray', height=100, width=200, font='Arial')
ChatLog.config(state=DISABLED)

scrollbar = Scrollbar(base, command=ChatLog.yview)
ChatLog['yscrollcommand'] = scrollbar.set

# Pack the scrollbar to the right side of the window
# scrollbar.pack(side=RIGHT, fill=Y)
# Pack the Text widget to fill the rest of the window
# ChatLog.pack()

AssignButton = Button(base, font=('Verdana', 14, 'bold'), text='Assign',
                      width=10, height=5, bd=0, bg='#32de97', activebackground='#3c9d9b', fg='#ffffff',
                      command=send)

InputBox = Text(base, bd=0, bg='gray', width=200, height=300, font='Arial')
# InputBox.bind_all("<Key>", on_key_release, '+')
InputBox.bind('<KeyRelease>', on_key_release, '+')

scrollbar.place(x=380, y=5, height=400)
InputBox.place(x=5, y=5, height=400, width=375)

ChatLog.place(x=5, y=410, height=85, width=280)
AssignButton.place(x=290, y=410, height=85, width=105)

base.mainloop()
