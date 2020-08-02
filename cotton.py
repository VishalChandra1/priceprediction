from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

cotton = Flask(__name__)

model1=pickle.load(open('model1.pkl','rb'))

@cotton.route('/')
def hello_world():
	return render_template("forest_fire.html")

@cotton.route('/predict',methods=['POST','GET'])
def predict:
	int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
   
if __name__ == '__main__':
    cotton.run(debug=True)


