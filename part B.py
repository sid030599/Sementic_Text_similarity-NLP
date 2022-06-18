from flask import Flask, request,jsonify
import pickle

from itsdangerous import json
app = Flask(__name__)
model = pickle.load(open('sts_model.h5', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def home():
    json_ = request.json
    prediction = model.predict([json_])
    prediction = round(prediction[0],1)
    return jsonify({'Similarity score': str(prediction)})



if __name__ == '__main__':
    app.run(debug=True)