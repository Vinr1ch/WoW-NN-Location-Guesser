"website"
#Load operation system library
import os

#website libraries
from flask import render_template
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

#Load math library
import numpy as np
import pickle
import os

#Load machine learning libraries
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.compat.v1.keras.backend import set_session
import tensorflow as tf
import cv2



Z = "Arenweald"
Y = "Bastion"
X = "Maldraxxus"
A = "Revendreth"

sampleZ = "static/Ardenweald.jpg"
sampleY = "static/Bastion.jpg"
sampleX = "static/Maldraxxus.jpg"
sampleA = "static/Revendreth.jpg"




UPLOAD_FOLDER = 'static/uploads'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# Machine Learning Model Filename
ML_MODEL_FILENAME = 'wowzone.model'


# Create the website object
app = Flask(__name__)

def load_model_from_file():
    #Set up the machine learning session

    myModel = load_model(ML_MODEL_FILENAME)
    myGraph = tf.compat.v1.get_default_graph()
    lb = pickle.loads(open('lb.pickle', "rb").read())
    return (myModel,lb)

#Try to allow only images
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#Define the view for the top level page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    #Initial webpage load
    if request.method == 'GET' :
        return render_template('index.html',myX=X,myY=Y,myA=A,myZ=Z,mySampleX=sampleX,mySampleY=sampleY,mySampleZ=sampleZ,mySampleA=sampleA)
    else: # if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser may also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # If it doesn't look like an image file
        if not allowed_file(file.filename):
            flash('I only accept files of type'+str(ALLOWED_EXTENSIONS))
            return redirect(request.url)
        #When the user uploads a file with good parameters
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    test_image = cv2.imread(UPLOAD_FOLDER+"/"+filename)
    #test_image = image.img_to_array(test_image)
    #test_image = np.expand_dims(test_image, axis=0)

    test_image = cv2.resize(test_image, (96, 96))
    test_image = test_image.astype("float") / 255.0
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    #mySession = app.config['SESSION']
    myModel = app.config['MODEL']
    lb = app.config['GRAPH']
    #with myGraph.as_default():
        #set_session(mySession)
    result = myModel.predict(test_image)[0]
    #proba = model.predict(image)[0]
    idx = np.argmax(result)
    label = lb.classes_[idx]
    print(label)

    image_src = "/"+UPLOAD_FOLDER +"/"+filename
    #if result[0] < 0.5 :
    answer = "<div class='col text-center'><img width='150' height='150' src='"+image_src+"' class='img-thumbnail' /><h4>guess:"+label+" "+str(result[idx] * 100)+ "%" + "</h4> </div><div class='col'></div><div class='w-100'></div>"
    #else:
    #answer = "<div class='col'></div><div class='col text-center'><img width='150' height='150' src='"+image_src+"' class='img-thumbnail' /><h4>guess:"+Y+" "+str(result[0])+"</h4></div><div class='w-100'></div>"
    results.append(answer)
    return render_template('index.html',myX=X,myY=Y,myA=A,myZ=Z,mySampleX=sampleX,mySampleY=sampleY,mySampleZ=sampleZ,mySampleA=sampleA,len=len(results),results=results)

def main():
    (myModel,lb) = load_model_from_file()

    app.config['SECRET_KEY'] = 'super secret key'

    #app.config['SESSION'] = mySession
    app.config['MODEL'] = myModel
    app.config['GRAPH'] = lb

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 #16MB upload limit
    app.run(port=9000, debug=False)

# Create a running list of results
results = []

#Launch everything
main()
