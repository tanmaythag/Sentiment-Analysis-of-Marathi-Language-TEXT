import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import pickle
import requests
import json
import sklearn

df = pd.read_csv('final_data - final_data.csv')

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_df=0.90,min_df=2,max_features=1000)
vect = vectorizer.fit_transform(df['token'])

vect_stemmed = vectorizer.fit_transform(df['token_stemmed'])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(vect,df['label'],random_state=7,test_size=0.25)

from sklearn.ensemble import ExtraTreesClassifier
bn=ExtraTreesClassifier()
bn.fit(x_train,y_train)
filename = 'finalized_model.sav'
pickle.dump(bn, open(filename, 'wb'))
 

 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(x_test)
print(result)
