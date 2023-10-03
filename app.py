import numpy as np
from flask import Flask, request, render_template
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer

IMAGE_FOLDER = os.path.join('static', 'img_pool')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

# Load the model
#model = pickle.load(open('finalized_model.sav','rb'))
def sentimentan(text):
    vectorizer = CountVectorizer()
    vect = vectorizer.fit_transform(['text'])
    model = pickle.load(open('finalized_model.sav','rb'))
    result = model.predict(vect)
    return result

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/result', methods = ['POST'])
def result():
   if request.method == 'POST':
        text = request.form['comment']
        re = sentimentan(text)
        if int(re)== 1:
            sentiment ='Positive'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Smiling_Emoji.png')
        elif int(re)==-1:
            sentiment ='Negative'
	        img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Sad_Emoji.png') 
        else:
            sentiment ='Neutral' 
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Neutral_Emoji.png') 
         
        return render_template("result.html", sentiment = sentiment, image=img_filename)

if __name__ == '__main__':
   app.run(debug = True)