import pickle
import pandas as pd
from flask import Flask, request, render_template

model = pickle.load(open('rf_model1.pkl', 'rb'))

app = Flask(__name__)


def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

@app.route("/")
def index():
    return render_template('index1.html')

@app.route("/form.html", methods=['GET', 'POST'])

def form():
    if request.method == 'GET':
        return render_template('form1.html')
    elif request.method == 'POST':
        form_inputs = request.form.to_dict()
        predictions = model.predict(pd.DataFrame(form_inputs, index=[0]).astype(float))
        #return_stmt = "Customer is having good credit. Will not be defaulter in future." if predictions==0 else "Customer is not having good credit. Customer will be defaulter in future."
        return render_template('form_results1.html', predictions=predictions)

if __name__ == "__main__":
    app.run(debug=True)