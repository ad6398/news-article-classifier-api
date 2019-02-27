from flask import Flask, request, jsonify
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import matplotlib.pyplot as plt
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def pre_process(text):
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    
    words = ""
    
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    return words

application = Flask(__name__)
model
@application.route('/predict', methods= ['POST'])
def predict():
    if model:
        query= request.json
        query= pd.DataFrame(query)
        query_body= query['email'].copy()

        query_body= query_body.apply(pre_process)
        vect = joblib.load('vectroizer.pkl')
        # selector = joblib.load('selector.pkl')

        # feat = vect.fit_transform(query_body))
        feat= vect.transform(query_body)
        prediciton = list(model.predict(feat))
        return jsonify({'prediciton': str(prediciton)})

    else:
        return({"error message": "trained model not found"})
    

if __name__=='__main__':
    

    model = joblib.load("news_article_clf_model.pkl")
    print("model loaded")
    
    application.debug= True
    application.run()
