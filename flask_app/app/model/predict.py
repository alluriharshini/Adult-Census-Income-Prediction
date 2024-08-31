import joblib
from .preprocess import derive_features,transform_features
import pandas as pd
# Load the trained model
model_path = 'app/model/model.pkl'
preprocessor_path='app/model/preprocessor.pkl'

def preprocess_user_input(data):
    df = pd.DataFrame(data, index=[0])
    print(data)
    return df
def load_model_and_preprocessor(model_path, preprocessor_path):
    clf = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return clf, preprocessor

def predict_salary(filepath):
    # Load and preprocess the new data
    with open("app/model/selected_features.txt", 'r') as file:
        # Read each line and strip leading/trailing whitespace
        selected_features = [line.strip() for line in file]


    df = preprocess_user_input(filepath)
    df=df.drop(['submit'],axis=1)
    columns_to_convert_int = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    df[columns_to_convert_int] = df[columns_to_convert_int].astype(int)

    columns_to_convert = ['education_num', 'marital_status', 'capital_gain', 
                      'capital_loss', 'hours_per_week']
    column_mapping = {col: col.replace('_', '-') for col in columns_to_convert}
    # Rename the specified columns using the dictionary
    df.rename(columns=column_mapping, inplace=True)

    df=derive_features(df)
    df=transform_features(df)
    
    clf, preprocessor = load_model_and_preprocessor(model_path, preprocessor_path)
    # Apply the preprocessor to the new data
    X_preprocessed = preprocessor.transform(df)

    feature_names = preprocessor.get_feature_names_out()  # Get all transformed feature names
    def clean_feature_names(feature_list):
        cleaned_names = [name.replace('num__', '').replace('cat__', '') for name in feature_list]
        return cleaned_names

    # Clean feature names
    cleaned_feature_names = clean_feature_names(feature_names)
    # Convert transformed data to DataFrame with appropriate feature names
    X_new_df = pd.DataFrame(X_preprocessed, columns=cleaned_feature_names)
    X_new_df=X_new_df[selected_features]
    # Make predictions
    predictions = clf.predict(X_new_df)
    return predictions