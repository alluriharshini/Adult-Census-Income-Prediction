import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.feature_selection import SelectKBest, chi2, f_classif, SequentialFeatureSelector
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.utils import resample  # Import for resampling
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score



# Step 1: Load data and preprocess
def load_data(filepath):
    """
    Load data from a CSV file.

    Input:
        filepath (str): Path to the CSV file.
    
    Output:
        df (pd.DataFrame): Loaded data as a pandas DataFrame.
    """
    df = pd.read_csv(filepath)
    df=df.sample(2000)
    df['salary']=df['salary'].map({' <=50K': 0, ' >50K': 1})

    return df
def derive_features(df):
    """
    Derive new features from existing ones.

    Input:
        df (pd.DataFrame): DataFrame containing the original features.
    
    Output:
        df (pd.DataFrame): DataFrame with new derived features.
    """
    # Example derived features: Ratio, interaction, and binning
    df['income_to_education'] = df['capital-gain'] / df['education-num']
    df['income_to_hours'] = df['capital-gain'] / df['hours-per-week']
    df['age_education_interaction'] = df['age'] * df['education-num']
    df['capital_interaction'] = df['capital-gain'] - df['capital-loss']
    df['age_bin'] = pd.cut(df['age'], bins=[0, 25, 50, 75, 100], labels=['Young', 'Middle-aged', 'Senior', 'Very Senior'])
    df['hours_bin'] = pd.cut(df['hours-per-week'], bins=[0, 30, 40, 60, 100], labels=['Part-time', 'Full-time', 'Overtime', 'Excessive'])

    # Replace rare categories in specific columns with 'Others'
    for col in ['country', 'workclass', 'occupation', 'education']:
        value_counts = df[col].value_counts()
        to_replace = value_counts[value_counts < 100].index
        df[col] = df[col].replace(to_replace, 'Others')
    
    return df
# Step 3: Handle Missing Data
def impute_data(df):
    """
    Handle missing data by replacing '?' with NaN and imputing missing values.

    Input:
        df (pd.DataFrame): DataFrame with missing values.
    
    Output:
        df (pd.DataFrame): DataFrame with imputed values.
    """
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Drop columns with too many missing values
    df = df.loc[:, df.isnull().mean() < 0.4]

    # Impute numerical columns with the median
    num_imputer = SimpleImputer(strategy='median')
    df[df.select_dtypes(include=[np.number]).columns] = num_imputer.fit_transform(df.select_dtypes(include=[np.number]))
    
    # Impute categorical columns with the most frequent value
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[df.select_dtypes(include=[object]).columns] = cat_imputer.fit_transform(df.select_dtypes(include=[object]))
    
    return df

# Step 4: Handle Outliers
def handle_outliers(df):
    """
    Cap outliers using the IQR method.

    Input:
        df (pd.DataFrame): DataFrame with potential outliers.
    
    Output:
        df (pd.DataFrame): DataFrame with capped outliers.
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    for column in df.select_dtypes(include=[np.number]).columns:
        df[column] = np.where(df[column] < Q1[column] - 1.5 * IQR[column], 
                              Q1[column] - 1.5 * IQR[column], df[column])
        df[column] = np.where(df[column] > Q3[column] + 1.5 * IQR[column], 
                              Q3[column] + 1.5 * IQR[column], df[column])
    
    return df
# Step 5: Feature Transformation
def transform_features(df):
    """
    Apply log transformation to skewed features.

    Input:
        df (pd.DataFrame): DataFrame with original features.
    
    Output:
        df (pd.DataFrame): DataFrame with transformed features.
    """
    df['log_capital_gain'] = np.log1p(df['capital-gain'])
    df['log_capital_loss'] = np.log1p(df['capital-loss'])
    df['log_hours_per_week'] = np.log1p(df['hours-per-week'])
    
    return df
# Step 6: Preprocessor Setup
def get_preprocessor(num_features, cat_features):
    """
    Creates a preprocessing pipeline for numerical and categorical features.
    
    Input:
        num_features (list): List of numerical feature names.
        cat_features (list): List of categorical feature names.
    
    Output:
        preprocessor (ColumnTransformer): Combined preprocessor for numerical and categorical features.
    """
    # Preprocessing pipelines for numerical and categorical features
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])  # Use sparse=False
    
    # Combine transformers into a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)])
    
    return preprocessor
def transform_features_with_preprocessor(preprocessor, X):
    """
    Transforms features using the provided preprocessor and returns the transformed DataFrame.
    
    Input:
        preprocessor (ColumnTransformer): Preprocessor pipeline for numerical and categorical features.
        X (pd.DataFrame): DataFrame with features to transform.
    
    Output:
        X_transformed (pd.DataFrame): Transformed features DataFrame with appropriate column names.
    """
    # Transform the features
    X_transformed = preprocessor.transform(X)
    
    # Extract feature names from the final step of each transformer
    num_feature_names = preprocessor.named_transformers_['num'].named_steps['scaler'].get_feature_names_out()
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
    
    # Combine the feature names
    feature_names = num_feature_names.tolist() + cat_feature_names.tolist()
    
    # Convert the transformed arrays back to DataFrames with proper feature names
    X_transformed = pd.DataFrame(X_transformed, columns=feature_names)
    
    return X_transformed


def preprocess_data(df, y_column):
    """
    Preprocess data by balancing, splitting, scaling, and encoding features.
    
    Input:
        df (pd.DataFrame): DataFrame containing the dataset.
        y_column (str): Name of the target variable column.
    
    Output:
        preprocessor (ColumnTransformer): Preprocessor object for future use.
        X_train (pd.DataFrame): Transformed training feature set.
        X_test (pd.DataFrame): Transformed test feature set.
        y_train (pd.Series): Training target values.
        y_test (pd.Series): Test target values.
    """
    # Splitting the dataset into features (X) and target (y)
    X = df.drop(columns=[y_column])
    y = df[y_column]


    # Combine X and y back into a single DataFrame for resampling
    df_combined = pd.concat([X, y], axis=1)

    # Separate majority and minority classes
    majority_class = df_combined[y_column].value_counts().idxmax()
    minority_class = df_combined[y_column].value_counts().idxmin()

    df_majority = df_combined[df_combined[y_column] == majority_class]
    df_minority = df_combined[df_combined[y_column] == minority_class]

    # Downsample the majority class to the size of the minority class
    df_majority_downsampled = resample(df_majority, 
                                       replace=False,    # Sample without replacement
                                       n_samples=len(df_minority),  # Match the number of minority class samples
                                       random_state=42)

    # Combine minority class with downsampled majority class
    df_balanced = pd.concat([df_minority, df_majority_downsampled])

    # Separate features and target variable again after balancing
    X_balanced = df_balanced.drop(columns=[y_column])
    y_balanced = df_balanced[y_column]

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

    # Identifying numeric and categorical features
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

    # Preprocessing for numeric data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())  # Use MinMaxScaler
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='error', drop='first'))  # Drop first category to prevent multicollinearity
    ])

    # Combining preprocessing for numeric and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit and transform X_train, and transform X_test
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Retrieve feature names after transformations
    numeric_feature_names = numeric_features.tolist()  # Convert to list for concatenation
    categorical_feature_names = list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features))

    # Combine numeric and categorical feature names
    feature_names = numeric_feature_names + categorical_feature_names

    print(f"Transformed X_train shape: {X_train_transformed.shape}, Feature names length: {len(feature_names)}")  # Debugging statement
    
    #X_train_transformed = X_train_transformed.toarray()  # if it's a sparse matrix
    #X_test_transformed = X_test_transformed.toarray()

    # Convert transformed data back to DataFrames with appropriate feature names
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    return preprocessor, X_train_df, X_test_df, y_train, y_test
# Step 8: VIF Calculation
def calculate_vif(X, selected_features):
    """
    Calculate the Variance Inflation Factor (VIF) for numerical features.

    Input:
        X (pd.DataFrame): DataFrame with selected features.
        selected_features (list): List of selected feature names.
    
    Output:
        vif_data (pd.DataFrame): DataFrame containing features and their VIF values.
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = selected_features
    vif_data["VIF"] = [variance_inflation_factor(X[selected_features].values, i) for i in range(len(selected_features))]
    return vif_data
# Step 9: Feature Selection
def feature_selection(X_processed, y):
    """
    Perform feature selection using ANOVA F-value, Chi-Square, Lasso, and Sequential Feature Selectors.
    
    Input:
        X_processed (pd.DataFrame): Processed features.
        y (pd.Series): Target variable.
    
    Output:
        final_selected_features (list): List of selected feature names.
    """
    # Filter methods: ANOVA F-value and Chi-Square
    selector_anova = SelectKBest(score_func=f_classif, k='all')
    selector_chi2 = SelectKBest(score_func=chi2, k='all')

    X_anova_selected = selector_anova.fit_transform(X_processed, y)
    X_chi2_selected = selector_chi2.fit_transform(X_processed, y)
    
    anova_scores = pd.Series(selector_anova.scores_, index=X_processed.columns)
    chi2_scores = pd.Series(selector_chi2.scores_, index=X_processed.columns)
    
    # Lasso Regularization
    lasso = LassoCV(cv=5)
    lasso.fit(X_processed, y)
    lasso_selected = pd.Series(lasso.coef_, index=X_processed.columns)
    
    # Sequential Feature Selectors: Forward and Backward
    logistic_model = LogisticRegression(max_iter=1000)
    
    # Forward Feature Selection
    sfs_forward = SequentialFeatureSelector(logistic_model, n_features_to_select=0.3, direction='forward')
    sfs_forward.fit(X_processed, y)
    forward_selected_features = X_processed.columns[sfs_forward.get_support()].tolist()
    
    # Backward Feature Selection
    sfs_backward = SequentialFeatureSelector(logistic_model, n_features_to_select=0.3, direction='backward')
    sfs_backward.fit(X_processed, y)
    backward_selected_features = X_processed.columns[sfs_backward.get_support()].tolist()
    
    # Aggregating selected features from different methods
    selected_features = set(anova_scores.nlargest(20).index) | set(chi2_scores.nlargest(20).index) | set(lasso_selected[lasso_selected != 0].index)
    selected_features = selected_features | set(forward_selected_features) | set(backward_selected_features)
    
    # Final selection based on frequency of occurrence across methods
    #feature_frequency = pd.Series(list(selected_features)).value_counts()
    #final_selected_features = feature_frequency[feature_frequency > 1].index.tolist()  # Keep features selected by more than one method
    
    # vif_data = calculate_vif(X_processed,selected_features)
    
    # while vif_data['VIF'].max() > 10:  # A threshold of 10 is often used
    #     max_vif_feature = vif_data.loc[vif_data['VIF'].idxmax(), 'feature']
    #     X_train = X_train.drop(columns=[max_vif_feature])
    #     vif_data = calculate_vif(X_processed,selected_features)
    # print(vif_data)
    
    return list(selected_features)
# Step 10: Model Training and Evaluation
def hyperparameter_tuning(X_train, y_train):
    models = {
    'LogisticRegression': {
        'model': LogisticRegression(),
        'params': {
            'solver': ['liblinear', 'lbfgs', 'saga'],
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'max_iter': [100, 200, 500]
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
    },
    'XGBClassifier': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'params': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'gamma': [0, 0.1, 0.2, 0.3],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
    },
    'SVM': {
        'model': SVC(),
        'params': {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
            'degree': [2, 3, 4]
        }
    },
    'KNeighborsClassifier': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
    },
    'GradientBoostingClassifier': {
        'model': GradientBoostingClassifier(),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10]
        }
    },
    'MLPClassifier': {
        'model': MLPClassifier(),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['tanh', 'relu', 'logistic'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    },
    'AdaBoostClassifier': {
        'model': AdaBoostClassifier(),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2, 1.0],
            'algorithm': ['SAMME', 'SAMME.R']
        }
    },
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    }
}
    best_models = {}

    for model_name, model_info in models.items():
        grid_search = GridSearchCV(model_info['model'], model_info['params'], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_models[model_name] = grid_search.best_estimator_
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")

    return best_models
# Step 11: Model Evaluation
def evaluate_models(models, X_test, y_test):
    """
    Evaluate multiple models on the test set and display performance metrics.

    Input:
        models (dict): Dictionary of model names and trained model objects.
        X_test (pd.DataFrame or np.ndarray): Test feature set.
        y_test (pd.Series or np.ndarray): True target values for the test set.
    
    Output:
        None
    """
    # Initialize a DataFrame to store the results
    results = pd.DataFrame(columns=['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])
    
    # Iterate through the models and evaluate each one
    for name, model in models.items():
        # Get predictions for the test set
        predictions = model.predict(X_test)
        
        # Calculate performance metrics
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')  # Using weighted average to handle class imbalance
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        
        # Append results to the DataFrame
        results = results.append({'Model': name, 'Accuracy': accuracy, 'F1 Score': f1,
                                  'Precision': precision, 'Recall': recall}, ignore_index=True)
        
        # Print classification report for each model
        print(f"\nClassification Report for {name}:\n")
        print(classification_report(y_test, predictions))
        
        # Plot confusion matrix in heatmap format
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title(f'Confusion Matrix for {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

    # Display the results table sorted by the F1 Score
    results = results.sort_values(by='F1 Score', ascending=False)
    print("\nSummary of Model Performance:\n", results)

    # Suggest the best model based on the highest F1 Score
    best_model_name = results.iloc[0]['Model']
    print(f"\nThe best model based on F1 Score is: {best_model_name}")
    return best_model_name

