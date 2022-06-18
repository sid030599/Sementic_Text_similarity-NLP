from flask import Flask, request, render_template
import pickle
app = Flask(__name__)
model = pickle.load(open('sts_model.h5', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text1 = request.form.get('text1')
        text2 = request.form.get('text2')
        print(text2, type(text1))
        prediction = model.predict((text1, text2))
        print(prediction)
    return render_template('index.html',prediction = prediction)


if __name__ == '__main__':
    app.run(debug=True)