from flask import Flask, render_template, request
import numpy as np
import pickle


model=pickle.load(open('HousePricePrediction.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def main():
    return render_template('newhtml.html')
   
@app.route('/predict', methods=['POST', 'GET'])
def home():
    num1=request.form['num1']
    num2=request.form['num2']
    num3=request.form['num3']
    num4=request.form['num4']
    num5=request.form['num5']

    arr=np.array([[num1,num2,num3,num4,num5]])
    pred=model.predict(arr)
    return render_template('new2html.html', data=pred)

if __name__=='__main__':
    app.run(debug=False)
