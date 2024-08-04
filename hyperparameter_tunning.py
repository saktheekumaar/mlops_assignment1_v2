import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
import matplotlib.pyplot as plt

# Load datasets
df_red = pd.read_csv('winequality-red.csv', sep=';')
df_white = pd.read_csv('winequality-white.csv', sep=';')

# Combine the datasets
df_red['type'] = 0
df_white['type'] = 1
df = pd.concat([df_red, df_white], axis=0)

# Separate features and target
X = df.drop('quality', axis=1)
y = df['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to create model
def create_model(optimizer='adam', kernel_initializer='uniform', neurons=64):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(neurons, kernel_initializer=kernel_initializer, activation='relu'))
    model.add(Dense(32, kernel_initializer=kernel_initializer, activation='relu'))
    model.add(Dense(16, kernel_initializer=kernel_initializer, activation='relu'))
    model.add(Dense(1, kernel_initializer=kernel_initializer))
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
    return model

# Objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
    kernel_initializer = trial.suggest_categorical('kernel_initializer', ['uniform', 'normal'])
    neurons = trial.suggest_int('neurons', 64, 128)
    batch_size = trial.suggest_int('batch_size', 10, 40)
    epochs = trial.suggest_int('epochs', 10, 20)

    # Create and compile model
    model = create_model(optimizer=optimizer, kernel_initializer=kernel_initializer, neurons=neurons)

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

    # Evaluate the model
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Create the Optuna study and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Print best parameters
print(f"Best parameters: {study.best_params}")

# Train and evaluate the model with the best parameters
best_params = study.best_params
best_model = create_model(optimizer=best_params['optimizer'], kernel_initializer=best_params['kernel_initializer'], neurons=best_params['neurons'])
best_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], validation_split=0.2, verbose=0)

# Evaluate the best model
y_pred = best_model.predict(X_test).flatten()
mse = mean_squared_error(y_test, y_pred)
print(f"Test Mean Squared Error: {mse:.4f}")

# Calculate accuracy score
y_pred_class = y_pred.round().astype(int)
test_accuracy = accuracy_score(y_test, y_pred_class)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot optimization history
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.show()

# Plot hyperparameter importances
optuna.visualization.matplotlib.plot_param_importances(study)
plt.show()

# Plot hyperparameter relationships
optuna.visualization.matplotlib.plot_parallel_coordinate(study)
plt.show()
# docker tag 088a7d070309:v2  purushothamkilari/wine-quality-detector:latest
