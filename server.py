from flask import Flask
import numpy as np
import nltk
import os
import time

import pickle

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras import optimizers

from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

TOKENIZER = "tokenizer.pickle"
MODEL = "model.json"
WEIGHTS_FILE = "best_acc_meer_model_weights.hdf5"

@app.route("/api/<phrase>", methods=["GET"])
def home(phrase):
    print(str(phrase))
    proba, label = test_prediction(str(phrase), load_model)
    print('PROBABILITY : {} | LABEL : {}'.format(proba, label))
    return 'PROBABILITY : {} | LABEL : {}'.format(proba, label)

def get_coefs(word, *arr): 
    return word, np.asarray(arr, dtype='float32')

def test_prediction(sent, model):

    if len(sent) == 0:
        print("No input entered - please enter a valid input")
        return

    max_features, embed_size, maxlen = 100000, 100, 150
    tokenizer_file = open(TOKENIZER,'rb')
    tokenizer = pickle.load(tokenizer_file)
    tokenizer_file.close()
    features = sent.lower()
    sentences = tokenizer.texts_to_sequences([features])
    padded_features = pad_sequences(sentences, maxlen=maxlen) #features
    label_list = loaded_model.predict(padded_features)
    print(label_list)
    # classes = ["Toxic",", Severely Toxic",", Obscene",", a Threat",", an Insult",", Identity Hate"]
    classes = ['Not Toxic', 'Toxic']
    print("The sentence : '{}', is classified as : \n".format(sent))
    returned_class = classes[1] if label_list[0][0]>=0.5 else classes[0]
    """
    for i in range(len(classes)):
        if predict_labels[i] == 1:
            print(classes[i],end=" ")
    """
    proba = 1-label_list[0][0] if label_list[0][0] < 0.5 else label_list[0][0] 
    return proba, returned_class
    

if __name__ == "__main__":
    print("LOADING MODEL>>>")
    
    json_file = open(MODEL, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(WEIGHTS_FILE)
    loaded_model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizers.Adam(clipnorm=1.))
    print("Loaded model from disk")
    print("Waiting for model to load and cache build")
    time.sleep(5)
    print("Running Test prediction")
    proba, label = test_prediction('I am hungry, you bitch. My hatred for you knows no bounds', model=load_model)
    print('PROBABILITY : {} | LABEL : {}'.format(proba, label))
    app.run(debug=False,host='0.0.0.0', port=7117, threaded=False)