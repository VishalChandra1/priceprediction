import pickle

def predictmethod():
	predicted_price =pickle.load(open('model1.pkl','rb'))
	return predicted_price