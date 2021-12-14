from flask import Flask, request, render_template
import model

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/recommend", methods=['POST'])
def predict():
    if request.method == 'POST':
        int_features = [x for x in request.form.values()]
        output=model.product_predict(int_features)
        return render_template('index.html', tables=[output.to_html(classes='data')], titles=output.columns.values)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    print('***App Started***')
    app.run(debug=True)
