# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np







# Load the Random Forest CLassifier model
filename = 'RFCL_Daibetes'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/homepage')
def homepage():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
	
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf/5, age]])
        my_prediction = classifier.predict(data)
        
        values = {}
        values[0] = preg
        values[1] = glucose
        values[2] = bp
        values[3] = st
        values[4] = insulin
        values[5] = bmi
        values[6] = dpf
        values[7] = age
        

        return render_template('result.html', prediction=my_prediction, accuracy = 92.50,values = values)

@app.route('/gmap')
def gmap():
    return render_template('map.html')


@app.route('/survey')
def survey():
    return render_template('survey.html')


if __name__ == '__main__':
	app.run(debug=True)