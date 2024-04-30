# %% Necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns


# %%
# Function to load and preprocess data
def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)

    df = df.dropna(axis=1)
    
    df = pd.get_dummies(df, columns=['diagnosis'], drop_first=True)

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    X = df_scaled.drop('diagnosis_M', axis=1)
    y = df_scaled['diagnosis_M']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# %%
# Function to define and train a machine learning model pipeline
def train_model(model_name, X_train, y_train, hyperparameter_grid):
    model_dict = {
        'Logistic Regression': Pipeline([('scaler', MinMaxScaler()), ('model', LogisticRegression())]),
        'Support Vector Machine': Pipeline([('scaler', MinMaxScaler()), ('model', SVC())]),
        'Decision Tree': Pipeline([('scaler', MinMaxScaler()), ('model', DecisionTreeClassifier())]),
        'Naive Bayes': Pipeline([('scaler', MinMaxScaler()), ('model', GaussianNB())]),
        'K-Nearest Neighbors': Pipeline([('scaler', MinMaxScaler()), ('model', KNeighborsClassifier())]),
        'Random Forest': Pipeline([('scaler', MinMaxScaler()), ('model', RandomForestClassifier())]),
        'Gradient Boosting': Pipeline([('scaler', MinMaxScaler()), ('model', GradientBoostingClassifier())]),
        'AdaBoost': Pipeline([('scaler', MinMaxScaler()), ('model', AdaBoostClassifier())]),
        'SGD Classifier': Pipeline([('scaler', MinMaxScaler()), ('model', SGDClassifier())])
    }

    model = model_dict[model_name].fit(X_train, y_train)

    # Grid search for hyperparameter tuning 
    if hyperparameter_grid is not None:
        grid_search = GridSearchCV(model, hyperparameter_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_

    return model


# %%
# Function to evaluate and visualize model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred) if hasattr(model, 'predict_proba') else None
    r2 = r2_score(y_test, y_pred) if hasattr(model, 'predict_proba') else None

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nModel: {model}")

    if mse is not None and r2 is not None:
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R^2 Score: {r2:.4f}")

    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


# %%
data_path = '/home/anis/Documents/Projects/myproject/Breast Cancer Wisconsin.csv'
X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)


# %% Hyperparameter grids 

# Logistic Regression
logistic_regression_grid = {
    'model__C': [1.0],
    'model__max_iter': [100, 500, 1000],
    'model__multi_class': ['auto'],
    'model__solver': ['lbfgs']
}

# Support Vector Machine
svm_grid = {
    'model__C': [0.1, 1, 10, 100],
    'model__gamma': [1, 0.1, 0.01, 0.001],
    'model__kernel': ['rbf', 'linear']
}

# Decision Tree
decision_tree_grid = {
    'model__max_depth': [None, 2, 4, 6, 8, 10],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

# Naive Bayes
naive_bayes_grid = {
    'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}

# K-Nearest Neighbors
knn_grid = {
    'model__n_neighbors': [3, 5, 11, 19],
    'model__weights': ['uniform', 'distance'],
    'model__metric': ['euclidean', 'manhattan']
}

# Random Forest
random_forest_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__criterion': ['gini', 'entropy'],
    'model__max_features': ['auto', 'sqrt', 'log2'],
    'model__max_depth': [None, 2, 4, 6, 8, 10],
    'model__min_samples_split': [2, 5, 10,],
    'model__min_samples_leaf': [1, 2, 4]
}

# Gradient Boosting
gradient_boosting_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__learning_rate': [0.01, 0.1, 1],
    'model__max_depth': [None, 2, 4, 6, 8, 10],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

# AdaBoost
adaboost_grid = {
    'model__n_estimators': [50, 100, 150, 200],
    'model__learning_rate': [0.01, 0.1, 1]
}

# SGD Classifier
sgd_grid = {
    'model__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
    'model__penalty': ['l2', 'l1', 'elasticnet'],
    'model__alpha': [0.0001, 0.001, 0.01, 0.1]
}


# %%
# Train and evaluate models 


# %% Logistic Regression
model = train_model('Logistic Regression', X_train, y_train, logistic_regression_grid)
evaluate_model(model, X_test, y_test)


# %% Support Vector Machine
model = train_model('Support Vector Machine', X_train, y_train, svm_grid)
evaluate_model(model, X_test, y_test)


# %% Decision Tree
model = train_model('Decision Tree', X_train, y_train, decision_tree_grid)
evaluate_model(model, X_test, y_test)


# %% Naive Bayes
model = train_model('Naive Bayes', X_train, y_train, naive_bayes_grid)
evaluate_model(model, X_test, y_test)


# %% K-Nearest Neighbors
model = train_model('K-Nearest Neighbors', X_train, y_train, knn_grid)
evaluate_model(model, X_test, y_test)


# %% Random Forest
model = train_model('Random Forest', X_train, y_train, random_forest_grid)
evaluate_model(model, X_test, y_test)


# %% Gradient Boosting
model = train_model('Gradient Boosting', X_train, y_train, gradient_boosting_grid)
evaluate_model(model, X_test, y_test)


# %% AdaBoost
model = train_model('AdaBoost', X_train, y_train, adaboost_grid)
evaluate_model(model, X_test, y_test)


# %% SGD Classifier
model = train_model('SGD Classifier', X_train, y_train, sgd_grid)
evaluate_model(model, X_test, y_test)

# %%