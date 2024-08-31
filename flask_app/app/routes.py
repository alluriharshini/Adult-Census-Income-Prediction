from flask import Blueprint, render_template, request, redirect, url_for, flash
from .forms import PredictForm
from .model.predict import predict_salary

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    form = PredictForm()
    if form.validate_on_submit():
        data = {key: request.form[key] for key in form.data.keys() if key != 'csrf_token'}
        prediction = predict_salary(data)
        return render_template('result.html', prediction=prediction)
    
    return render_template('index.html', form=form)

@main.route('/result')
def result():
    return render_template('result.html')
