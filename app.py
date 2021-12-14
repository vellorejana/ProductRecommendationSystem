from flask import Flask, request, render_template
import model
import pandas as pd

app = Flask(__name__)

user=pd.read_csv('./data/userdf.csv',usecols=['reviews_username'])
username = user.reviews_username.to_list()
username.append('<select user>')


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', username =username)


@app.route("/recommend", methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        int_features = [x for x in request.form.values()]
        output=model.product_predict(int_features)
        return render_template('index.html', tables=[output.to_html(classes='data')], titles=output.columns.values, username =username)
    elif request.method == 'GET':
		return render_template('index.html', username =username)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    print('***App Started***')
    app.run(debug=True)
