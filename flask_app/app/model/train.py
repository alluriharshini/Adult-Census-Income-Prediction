import pandas as pd
from preprocess import derive_features,feature_selection,evaluate_models,impute_data,handle_outliers,preprocess_data,hyperparameter_tuning,transform_features,calculate_vif,load_data
import warnings
import joblib
warnings.filterwarnings('ignore')


def train_and_save_model(filepath):
    df = load_data(filepath)

    y = df['salary']  # Extract target variable
    df = df.drop('salary', axis=1)  # Remove target variable from the DataFrame

    # Step 2: Perform feature engineering steps without 'salary'
    df = derive_features(df)
    df = impute_data(df)
    df = handle_outliers(df)
    df = transform_features(df)

    # Step 3: Add 'salary' column back to the DataFrame
    df['salary'] = y

    # Preprocess features and split data
    preprocessor, X_train, X_test, y_train, y_test = preprocess_data(df, 'salary')


    # Perform feature selection
    selected_features = feature_selection(X_train, y_train)
    
    X_train=X_train[selected_features]
    # Calculate VIF to check multicollinearity
    vif_data = calculate_vif(X_train, selected_features)
    print("Variance Inflation Factor (VIF) for selected features:")

    while vif_data['VIF'].max() > 10:  # A threshold of 10 is often used
        max_vif_feature = vif_data.loc[vif_data['VIF'].idxmax(), 'Feature']
        print(max_vif_feature)
        X_train = X_train.drop(columns=[max_vif_feature])
        selected_features.remove(max_vif_feature)
        vif_data = calculate_vif(X_train,selected_features)
    
    # Train and evaluate models on the selected features
    best_models = hyperparameter_tuning(X_train, y_train)
    X_test=X_test[selected_features]
    print(X_train.columns,"selected_feature final")
    best_model_name=evaluate_models(best_models, X_test, y_test)
    clf=best_models[best_model_name]
    
    # Save the model and the preprocessor
    joblib.dump(clf, 'app/model/model.pkl')
    joblib.dump(preprocessor, 'app/model/preprocessor.pkl')

    # Write the list to a text file
    with open('app/model/selected_features.txt', 'w') as file:
        for item in X_train.columns:
            file.write(f"{item}\n")

if __name__ == "__main__":
    train_and_save_model('data/adult.csv')
