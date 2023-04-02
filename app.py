from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    # Feature and labels
    # df['v2'] = df['v2'].rename(df['message'], inplace=True)
    df['label'] = df['v1'].map({'ham': 0, 'spam': 1})
    x = df['v2']
    y = df['label']
    
    # Extract Feature with Countvectoriser
    cv = CountVectorizer()
    X = cv.fit_transform(x)
    
    # Spilitting the dataset for Train and Test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    # Initialize the Model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    model.score(X_test, y_test)
    
    # Alternative Usage of Saved Medel
    # joblib.dump(model, 'Spam_Detection_Model.sav')
    # loaded_model = joblib.load('Spam_Detection_Model.sav)
    # result = loaded_model.score(X_test, y_test)
    # print(result)
    
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = model.predict(vect)
    
    return render_template('result.html', prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)