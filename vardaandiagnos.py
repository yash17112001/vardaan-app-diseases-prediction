import pickle
from flask import Flask, jsonify, request
import sklearn
hrtd = pickle.load(open("Heartdisease.pkl", 'rb'))
hrtd_std = pickle.load(open("Heartdisease_std.pkl", 'rb'))

pcos = pickle.load(open("Pcos.pkl", 'rb'))
pcos_std = pickle.load(open("Pcos_transform.pkl", 'rb'))

diabetes = pickle.load(open("Diabetes.pkl", 'rb'))
diabetes_std = pickle.load(open("Diabetes_transform.pkl", 'rb'))

hrts = pickle.load(open("Heartstroke.pkl", 'rb'))
hrts_std = pickle.load(open("Heartstroke_transform.pkl", 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "Welcome"


@app.route('/predictheartdiseases')
def heartdpred():
    sex = request.form.get('sex')
    systol = request.form.get('systol')
    bsugarbool = request.form.get('bsugarbool')
    restecg = request.form.get('restecg')
    thalachh = request.form.get('thalachh')
    oldpeak = request.form.get('oldpeak')
    spo2 = request.form.get('spo2')

    result = hrtd.predict(hrtd_std.transform([[sex, systol, bsugarbool, restecg, thalachh, oldpeak, spo2]]))[0]
    return jsonify(str(result))


@app.route('/predictpcos')
def pcospred():
    import sklearn
    age = request.form.get('age')
    bmi = request.form.get('bmi')
    hairgrowth = request.form.get('hairgrowth')
    skindar = request.form.get('skindar')
    pimples = request.form.get('pimples')
    fastfood = request.form.get('fastfood')
    systol = request.form.get('systol')
    diastol = request.form.get('diastol')

    return jsonify(str(pcos.predict(pcos_std.transform([[age, bmi, hairgrowth, skindar, pimples, fastfood, systol, diastol]]))[0]))


@app.route('/predictdiabetes')
def diabetespred():
    age = request.form.get('age')
    bmi = request.form.get('bmi')
    diastol = request.form.get('diastol')
    bsugar = request.form.get('bsugar')
    return jsonify(str(diabetes.predict(diabetes_std.transform([[bsugar, diastol, bmi, age]]))[0]))

@app.route('/predictstroke')
def strokepred():
    age = request.form.get('age')
    hypertension = request.form.get('hypertension')
    heart_disease = request.form.get('heart_diseases')
    bsugar = request.form.get('bsugar')
    bmi = request.form.get('bmi')
    smoking_status = request.form.get('smoking_status')

    return jsonify(str(hrts.predict(hrts_std.transform([[age, hypertension, heart_disease, bsugar, bmi, smoking_status]]))[0]))


if __name__ == '__main__':
    app.run(debug=True)

