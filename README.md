# Adult-Census-Income-Prediction
```bash
flask_app/
│
├── app/
│   ├── __init__.py
│   ├── routes.py
│   ├── forms.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── preprocess.py
│   │   ├── predict.py
│   │   └── model.pkl
|   |   |__ preprocessor.pkl
|   |   |__ selected_features.txt
│   ├── templates/
│   │   ├── index.html
│   │   ├── predict.html
│   │   └── result.html
│   └── static/
│       ├── css/
│         ├── styles.css
│
├── data/
│   ├── adult.csv
│
├── requirements.txt
└── run.py
```
To execute the Flask application:
1. Navigate to the root of the project.
2. Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model by running:
   ```bash
   python app/model/train.py
   ```
4. Start the Flask server:
   ```bash
   python run.py
