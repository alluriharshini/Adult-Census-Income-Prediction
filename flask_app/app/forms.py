from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, FloatField, SubmitField
from wtforms.validators import DataRequired, NumberRange

class PredictForm(FlaskForm):
    age = FloatField('Age', validators=[DataRequired(), NumberRange(min=0)])
    fnlwgt = FloatField('fnlwgt', validators=[DataRequired(), NumberRange(min=0)])
    workclass = SelectField('Workclass', choices = [
    (' Private', ' Private'), 
    (' Self-emp-not-inc', ' Self-emp-not-inc'), 
    (' Local-gov', ' Local-gov'), 
    (' State-gov', ' State-gov'), 
    (' Self-emp-inc', ' Self-emp-inc'), 
    (' Federal-gov', ' Federal-gov'), 
    (' Without-pay', ' Without-pay'), 
    (' Never-worked', ' Never-worked')
]
)
        
    education = SelectField('Education', choices = [
    (' HS-grad', ' HS-grad'),
    (' Some-college', ' Some-college'),
    (' Bachelors', ' Bachelors'),
    (' Masters', ' Masters'),
    (' Assoc-voc', ' Assoc-voc'),
    (' 11th', ' 11th'),
    (' Assoc-acdm', ' Assoc-acdm'),
    (' 10th', ' 10th'),
    (' 7th-8th', ' 7th-8th'),
    (' Prof-school', ' Prof-school'),
    (' 9th', ' 9th'),
    (' 12th', ' 12th'),
    (' Doctorate', ' Doctorate'),
    (' 5th-6th', ' 5th-6th'),
    (' 1st-4th', ' 1st-4th'),
    (' Preschool', ' Preschool')
])
    
    education_num = FloatField('Education Num', validators=[DataRequired(), NumberRange(min=0)])
    
    marital_status = SelectField('Marital Status', choices = [
    (' Married-civ-spouse', ' Married-civ-spouse'),
    (' Never-married', ' Never-married'),
    (' Divorced', ' Divorced'),
    (' Separated', ' Separated'),
    (' Widowed', ' Widowed'),
    (' Married-spouse-absent', ' Married-spouse-absent'),
    (' Married-AF-spouse', ' Married-AF-spouse')
])
    
    occupation = SelectField(' Occupation', choices = [
    (' Prof-specialty', ' Prof-specialty'),
    (' Craft-repair', ' Craft-repair'),
    (' Exec-managerial', ' Exec-managerial'),
    (' Adm-clerical', ' Adm-clerical'),
    (' Sales', ' Sales'),
    (' Other-service', ' Other-service'),
    (' Machine-op-inspct', ' Machine-op-inspct'),
    (' Transport-moving', ' Transport-moving'),
    (' Handlers-cleaners', ' Handlers-cleaners'),
    (' Farming-fishing', ' Farming-fishing'),
    (' Tech-support', ' Tech-support'),
    (' Protective-serv', ' Protective-serv'),
    (' Priv-house-serv', ' Priv-house-serv'),
    (' Armed-Forces', ' Armed-Forces')
])
    
    relationship = SelectField('Relationship', choices = [
    (' Husband', ' Husband'),
    (' Not-in-family', ' Not-in-family'),
    (' Own-child', ' Own-child'),
    (' Unmarried', ' Unmarried'),
    (' Wife', ' Wife'),
    (' Other-relative', ' Other-relative')
])
    
    race = SelectField('Race', choices = [
    (' White', ' White'),
    (' Black', ' Black'),
    (' Asian-Pac-Islander', ' Asian-Pac-Islander'),
    (' Amer-Indian-Eskimo', ' Amer-Indian-Eskimo'),
    (' Other', ' Other')
]
)
    
    sex = SelectField('Sex', choices=[(' Male', ' Male'), (' Female', ' Female')])
    
    capital_gain = FloatField('Capital Gain', validators=[DataRequired(), NumberRange(min=0)])
    capital_loss = FloatField('Capital Loss', validators=[DataRequired(), NumberRange(min=0)])
    hours_per_week = FloatField('Hours per Week', validators=[DataRequired(), NumberRange(min=0)])
    
    country = SelectField('Country',choices = [
    (' United-States', ' United-States'),
    (' Mexico', ' Mexico'),
    (' Philippines', ' Philippines'),
    (' Germany', ' Germany'),
    (' Canada', ' Canada'),
    (' Puerto-Rico', ' Puerto-Rico'),
    (' El-Salvador', ' El-Salvador'),
    (' India', ' India'),
    (' Cuba', ' Cuba'),
    (' England', ' England'),
    (' Jamaica', ' Jamaica'),
    (' South', ' South'),
    (' China', ' China'),
    (' Italy', ' Italy'),
    (' Dominican-Republic', ' Dominican-Republic'),
    (' Vietnam', ' Vietnam'),
    (' Guatemala', ' Guatemala'),
    (' Japan', ' Japan'),
    (' Poland', ' Poland'),
    (' Columbia', ' Columbia'),
    (' Taiwan', ' Taiwan'),
    (' Haiti', ' Haiti'),
    (' Iran', ' Iran'),
    (' Portugal', ' Portugal'),
    (' Nicaragua', ' Nicaragua'),
    (' Peru', ' Peru'),
    (' France', ' France'),
    (' Greece', ' Greece'),
    (' Ecuador', ' Ecuador'),
    (' Ireland', ' Ireland'),
    (' Hong', ' Hong'),
    (' Cambodia', ' Cambodia'),
    (' Trinadad&Tobago', ' Trinadad&Tobago'),
    (' Laos', ' Laos'),
    (' Thailand', ' Thailand'),
    (' Yugoslavia', ' Yugoslavia'),
    (' Outlying-US(Guam-USVI-etc)', ' Outlying-US(Guam-USVI-etc)'),
    (' Honduras', ' Honduras'),
    (' Hungary', ' Hungary'),
    (' Scotland', ' Scotland'),
    (' Holand-Netherlands', ' Holand-Netherlands')
]
)
        
    submit = SubmitField('Predict')