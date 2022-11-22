import pickle
from flask import  Flask,request,app,jsonify,url_for,render_template
import pandas as pd
import numpy as np
from sklearn.linear_model._base import _preprocess_data

app=Flask(__name__, template_folder='templates')
## Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def template():
    return render_template('template.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data) 
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__" :
    app.run(debug=True)

