# Importing essential libraries
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util 
import numpy as np
import pandas as pd
import pickle

model = SentenceTransformer('stsb-roberta-large')
# filename = 'similarity.pkl'
# model = pickle.load(open(filename, 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('sample.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
    	sentence1 = request.form['message1']
    	sentence2 = request.form['message2']
    	embedding1 = model.encode(sentence1, convert_to_tensor=True)
    	embedding2 = model.encode(sentence2, convert_to_tensor=True)
    	cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
    	return render_template('sample.html', cosine_score=cosine_scores.item())

if __name__ == '__main__':
	app.run(debug=False)