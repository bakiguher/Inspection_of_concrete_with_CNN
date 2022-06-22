import os
import numpy as np
from util import base64_to_pil
from itertools import islice


from flask import Flask, request, render_template,  jsonify, redirect
import tensorflow as tf
from tensorflow.python.keras.models import model_from_json
from tensorflow.keras.preprocessing import image


# model for mole class
Modelmk_json = "./model/modelbm.json"
Modelmk_weigths = "./model/modelbm.h5"

IMAGEDIM=64


app = Flask(__name__)


def take(n, iterable):
    "Return first n items of the iterable as a dict"
    return dict(islice(iterable, n))


def get_model(modeljson, weights):
    '''
    Function to load saved model and weights 
    '''
    model_json = open(modeljson, 'r')
    loaded_model_json = model_json.read()
    model_json.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights)
    return model


def model_predict(img: image, model, dima: int, dimb: int):
    '''
    Get the image data and return predictions
    '''
    img = img.resize((dima, dimb))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255
    preds = model.predict(x)

    return preds

# initialize models
modelmk = get_model(Modelmk_json, Modelmk_weigths)

@app.route('/', methods=['GET'])
def index():
    '''
    main page
    '''
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    predict function, loads the model calls the prediction and returns highest predicted 2 mole classes
    '''
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Make predictions
        prediction = model_predict(img, modelmk, IMAGEDIM, IMAGEDIM)[0][0]
        
        pred=prediction *100
        pred=pred.round(1)
        
        pred=int(pred)
        
        result=""
        prob=""
        if pred >=50 :
            result='%' + str(pred) + " No Cracks"
            #prob= 'Proabability : % ' + str(pred)
        else:
            result='%'+ str(100-pred) +" Cracked"
            #prob ='Proabability : % ' + str(100-pred)

        return jsonify(result=result, predresult=prob)
    return None


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
