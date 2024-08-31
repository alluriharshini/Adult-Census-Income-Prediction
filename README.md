# Adult-Census-Income-Prediction
EDA.ipynb: "Perform exploratory data analysis (EDA) to understand data distribution, identify patterns, and detect anomalies."

Model_training_testing.ipynb: "Train, validate, and test machine learning models with hyperparameter tuning, feature selection, and removal of multicollinearity to optimize salary prediction."

```bash
flask_app/
│
├── app/
│   ├── __init__.py             # Initializes the Flask application
│   ├── routes.py               # Contains the routes/endpoints for the app
│   ├── forms.py                # Defines forms using Flask-WTF
│   ├── model/
│   │   ├── __init__.py         # Initializes the model sub-package
│   │   ├── train.py            # Contains the training logic for the model
│   │   ├── preprocess.py       # Handles data preprocessing
│   │   ├── predict.py          # Handles predictions using the model
│   │   ├── model.pkl           # Serialized model file (pickle)
│   │   ├── preprocessor.pkl    # Serialized preprocessor (pickle)
│   │   └── selected_features.txt # Text file listing selected features
│   ├── templates/
│   │   ├── index.html          # HTML for the home page
│   │   ├── predict.html        # HTML for the prediction form
│   │   └── result.html         # HTML for displaying results
│   └── static/
│       └── css/
│           └── styles.css      # CSS file for styling
│
├── data/
│   └── adult.csv               # Dataset file
│
├── requirements.txt            # Lists Python dependencies
└── run.py                      # Main entry point to run the Flask app

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
