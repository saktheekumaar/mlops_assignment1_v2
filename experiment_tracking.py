import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df_red = pd.read_csv('winequality-red.csv', sep=';')
df_white = pd.read_csv('winequality-white.csv', sep=';')

# Combine the red and white wine datasets
df_red['type'] = 0
df_white['type'] = 1
df = pd.concat([df_red, df_white], axis=0)

data=df

# Define features and target
X = data.drop('quality', axis=1)
y = data['quality']
y = (y >= 7).astype(int)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Start MLflow experiment
mlflow.start_run()

# Logistic Regression Experiment
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)

# Log Logistic Regression experiment
mlflow.log_param("model1 type", "Logistic Regression")
mlflow.log_param("model1 max_iter", 200)
mlflow.log_metric("model1 accuracy", lr_accuracy)

# Random Forest Experiment
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

# Log Random Forest experiment
mlflow.log_param("model2 type", "Random Forest")
mlflow.log_param("model2 n_estimators", 100)
mlflow.log_metric("model2 accuracy", rf_accuracy)

# SVM Experiment
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)

# Log SVM experiment
mlflow.log_param("model3 type", "SVM")
mlflow.log_metric("model3 accuracy", svm_accuracy)

mlflow.end_run()

print(f"Logistic Regression Accuracy: {lr_accuracy:.2f}")
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(f"SVM Accuracy: {svm_accuracy:.2f}")
