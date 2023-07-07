from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)

with open('kd.pkl','rb') as file:
 model = pickle.load(file)

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
         input_features = [float(x) for x in request.form.values()]
         features_value = [np.array(input_features)]
         features_name = ['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane']
         df = pd.DataFrame(features_value,columns=features_name)
         prediction = model.predict(df)
         return render_template('Result.html',prediction=prediction)
   
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True,port=5000)
