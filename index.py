from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

# prediction function 
@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        gender = float(request.form['gender'])
        work = float(request.form['work'])
        age = float(request.form['age'])
        married = float(request.form['married'])
        IPS1 = float(request.form['ips1'])
        IPS2 = float(request.form['ips2'])
        IPS3 = float(request.form['ips3'])
        IPS4 = float(request.form['ips4'])
        IPS5 = float(request.form['ips5'])
        IPS6 = float(request.form['ips6'])
        IPS7 = float(request.form['ips7'])
        IPS8 = float(request.form['ips8'])
        IPK = np.average([IPS1, IPS2, IPS3, IPS4, IPS5, IPS6, IPS7, IPS8])
        data = [[gender, work, age, married, IPS1, IPS2, IPS3, IPS4, IPS5, IPS6, IPS7, IPS8, IPK]]
        
        model = joblib.load('kelulusan_model.pkl')
        predict = model.predict(data)
                
        # result = ValuePredictor(to_predict_list)        
        if predict == 0:
            hasil = "Anda terprediksi terlambat lulus"
        elif predict == 1:
            hasil = "Anda terprediksi lulus tepat Waktu"
        else:
            hasil = "ERROR"          
        return render_template("result.html", prediction = hasil)