# flask,pandas,scikit-learn,pipe
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
data=pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route("/")
def index():
    locations= sorted(data['region'].unique())
    apts=sorted(data['type'].unique())
    return render_template('index.html',locations=locations,apts=apts)

@app.route("/predict",methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    apt= request.form.get('apt')
    total_sqft = float(request.form.get('total_sqft'))
    print(location,bhk,apt,total_sqft)
    input = pd.DataFrame([[bhk,apt,total_sqft,location]],columns=['bhk','type','area','region'])
    prediction = pipe.predict(input)[0]

    return str(prediction)

if __name__ == '__main__':
        app.run(debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
