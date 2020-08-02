from flask import Flask,request, redirect, render_template,Response
from predict import predictmethod
import flask
app = Flask(__name__)



@app.route('/',methods=['POST','GET'])
def hello_world():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	months=int(flask.request.form['Months'])
	#print(months)
	predicted_price=predictmethod()
	predicted_price=predicted_price[0:months]
	return render_template('ans.html',predicted_price=predicted_price)
   
if __name__ == '__main__':
    app.run(debug=True)


