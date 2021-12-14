from flask import Flask, request, render_template
import preprocess

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/recommend", methods=['POST'])
def predict():
    if request.method == 'POST':
        int_features = [x for x in request.form.values()]
        output=preprocess.product_predict(int_features)
        return render_template('index.html', tables=[output.to_html(classes='data')], titles=output.columns.values)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    print('***App Started***')
    app.listen(process.env.PORT || 3000, function(){
    console.log("Express server listening on port %d in %s mode", this.address().port, app.settings.env);
    });
