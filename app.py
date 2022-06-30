import os
import numpy as np
from util import base64_to_pil



from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.python.keras.models import model_from_json
from tensorflow.keras.preprocessing import image


# model for mole class
MODEL_JSON = "./model/modelbm.json"
MODEL_WEIGHTS = "./model/modelbm.h5"

# Dimensions of the image
IMAGEDIM = 64


app = Flask(__name__)


def get_model(modeljson, weights):
    """
    Function to load saved model and weights
     Args:
      modeljson (json file) - json structure of the model
      weights (binary) - contains the weights of the model

    """
    model_json = open(modeljson, "r")
    loaded_model_json = model_json.read()
    model_json.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights)
    return model


def model_predict(img: image, model, dima: int, dimb: int):
    """
    Get the image data and return prediction
    Args:
     img (image) : image data
     model (tensorflow model ) : model
     dima (integer) : dimension x of the image
     dimb (integer) : dimension y of the image
    """
    img = img.resize((dima, dimb))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255
    preds = model.predict(x)

    return preds


# initialize models
modelm_cement = get_model(MODEL_JSON, MODEL_WEIGHTS)


@app.route("/", methods=["GET"])
def index():
    """
    main page
    """
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    predict function, calls the prediction and returns a rounded value betwwen 0 or 1
    """
    if request.method == "POST":
        # Get the image from post request
        img = base64_to_pil(request.json)
        #print(type(img))
        #read_image(img)
        # Make predictions
        prediction = model_predict(img, modelm_cement, IMAGEDIM, IMAGEDIM)[0][0]

        pred = prediction * 100
        pred = pred.round(1)

        pred = int(pred)

        result = ""
        prob = ""
        if pred >= 50:
            result = "%" + str(pred) + " Cracked"
        else:
            result = "%" + str(100 - pred) + " Clean"

        return jsonify(result=result, predresult=prob)
    return None


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
