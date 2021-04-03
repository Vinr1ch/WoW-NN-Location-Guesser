#used to convert model to .js
from tensorflow.keras.models import load_model
import tensorflowjs as tfjs
import argparse
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
args = vars(ap.parse_args())

    # load the image


model = load_model(args["model"])
tfjs.converters.save_keras_model(model, "my_model.json")
model.save_weights("model.h5")
