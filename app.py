import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
AdaBoostmodel=pickle.load(open('AdaBoostmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(type(data))
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=AdaBoostmodel.predict(new_data)
    arr = np.array(output, dtype=np.int32)
    arr_ints = arr.tolist()
    print(arr_ints[0])
    return jsonify(arr_ints[0])


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=AdaBoostmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="Customer belongs to : Cluster --> {}".format(output))



if __name__=="__main__":
    app.run(debug=True)
   
     