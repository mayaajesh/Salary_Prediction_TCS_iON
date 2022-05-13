from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
   value = [float(x) for x in request.form.values()]
   value=[np.array(value)]
   prediction=model.predict(value)
   output=prediction[0]
   return render_template ('result.html',prediction_text="The predicted Salary of this employee is: {} ".format(output))
if __name__=='__main__':
    app.run(port=8000)