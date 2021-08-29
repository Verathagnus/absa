from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.models import load_model
import tensorflow as tf
import warnings
import pandas as pd
import numpy as np
import json
import re
import pickle
import string
import sys
import getopt

"""# Process"""

# Spacy

# Other

warnings.filterwarnings('ignore')

# Keras


def remove_punc(aspect):
    punc = string.punctuation
    temp = ""
    for i in aspect:
        if i not in punc:
            temp += i
    return temp

import io
import json

from flask import Flask, jsonify, request

app = Flask(__name__)

def get_prediction(model, text, aspect):
  df = pd.DataFrame({'text':[text], 'aspect':[aspect]})
  test_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(df.text))
  test_tokenized2 = pd.DataFrame(tokenizer2.texts_to_matrix(df.aspect))
  test_tokenized3 = pd.concat([test_tokenized, test_tokenized2], axis=1)
  # print(test_tokenized3)
  if model == 'dl':
    return label_encoder.inverse_transform(np.argmax(absa_model.predict(test_tokenized3), axis=-1))
  elif model in ("lr", "knn", "dt", "svc"):
    return absa_model.predict(test_tokenized3)
  else:
    return 'Error'
def load_model_encoders(model, modelfile):
  global absa_model
  global tokenizer
  global tokenizer2
  global label_encoder
  model_loc = "./models/"
  text_tok = model_loc+modelfile+'text_tokenizer_'+model
  with open(text_tok, 'rb') as handle:
    tokenizer = pickle.load(handle)
  aspect_tok = model_loc+modelfile+'aspect_tokenizer_'+model
  with open(aspect_tok, 'rb') as handle:
    tokenizer2 = pickle.load(handle)
  label_en = model_loc+modelfile+'label_encoder_'+model
  with open(label_en, 'rb') as handle:
    label_encoder = pickle.load(handle)
  
  if model == "dl":
    model_name = model_loc+modelfile+'dl'
    absa_model = load_model(model_name)
    return
  if model == "lr":
    pkl_filename = model_loc+modelfile+'lr.pkl'
    with open(pkl_filename, 'rb') as file:
      absa_model = pickle.load(file)
    return
  if model == "knn":
    pkl_filename = model_loc+modelfile+'knn.pkl'
    with open(pkl_filename, 'rb') as file:
      absa_model = pickle.load(file)
    return
  if model == "dt":
    pkl_filename = model_loc+modelfile+'dt.pkl'
    with open(pkl_filename, 'rb') as file:
      absa_model = pickle.load(file)
    return
  if model == "svc":
    pkl_filename = model_loc+modelfile+'svc.pkl'
    with open(pkl_filename, 'rb') as file:
      absa_model = pickle.load(file)
    return

@app.route('/', methods=['GET'])
def home():
   return jsonify({'model': 'MODEL_TYPE', 'modelfile': 'MODEL_FILE_NAME', 'text': 'TEXT', 'aspect': 'ASPECT'})

@app.route('/predict', methods=['GET'])
def predict():
    if request.method == 'GET':
        # data = request.get_json()     # status code
        # return request.args
        # return jsonify({'data': data})
        if ['aspect', 'model', 'modelname', 'text'] != sorted(list(request.args.keys())):
          return jsonify({'model': 'MODEL_TYPE', 'modelfile': 'MODEL_FILE_NAME', 'text': 'TEXT', 'aspect': 'ASPECT'})
        model = str(request.args.get('model'))
        if request.args.get('modelfile') == None:
          modelfile = ""
        else:
          modelfile = str(request.args.get('modelfile'))
        text = str(request.args.get('text'))
        aspect = str(request.args.get('aspect'))
        print(text, aspect)
        # if model != cache_model and modelfile != cache_modelfile: 
        load_model_encoders(model, modelfile)
          # cache_model = model
          # cache_modelfile = modelfile
        
        sentiment = get_prediction(model, text, aspect).tolist()[0]
        return jsonify({'sentiment': sentiment})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)