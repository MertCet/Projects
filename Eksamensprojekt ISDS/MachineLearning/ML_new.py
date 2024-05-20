# Linear Regression on Property Valuation Data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, validation_curve
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r'W:\OneDrive - University of Copenhagen\DataBackup\AfleveringsMappe\DataPreparation\merged_dataset_clean.csv', delimiter=';', decimal=',')

# Drop rows with missing values
df.dropna(inplace=True)

# Data Exploration
print(df.info())
print(df.describe())

# Handling Time Series Data
df['ValuationChangeDate'] = pd.to_datetime(df['ValuationChangeDate'])
df['lag_1_year'] = df.groupby('propertyHouseSize')['PropertyValuationAmount'].shift(1)
df['lag_2_years'] = df.groupby('propertyHouseSize')['PropertyValuationAmount'].shift(2)
df['rolling_mean_2_years'] = df.groupby('propertyHouseSize')['PropertyValuationAmount'].rolling(window=2).mean().reset_index(0, drop=True)
df['rolling_std_2_years'] = df.groupby('propertyHouseSize')['PropertyValuationAmount'].rolling(window=2).std().reset_index(0, drop=True)

# Model Training and Evaluation
df.dropna(subset=['lag_1_year', 'lag_2_years', 'rolling_mean_2_years', 'rolling_std_2_years'], inplace=True)
features = ['propertyHouseSize', 'propertySizeTotal', 'numberToilets', 'sizeGarage', 'MunicipalitySize', 'PropertyTaxRate', 'MunicipalityAverageIncome', 'MunicipalityPersonsCount', 'MunicipalityUnemployment', 'rolling_std_2_years']
X = df[features]
y = df['PropertyValuationAmount']

# Drop rows with missing values in the features and target variable
df.dropna(subset=features + ['PropertyValuationAmount'], inplace=True)
# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate the correlation matrix to examine the relationship between features and the target variable
correlation_matrix = df[features + ['PropertyValuationAmount']].corr()

# Get the correlation of all features with the target variable 'PropertyValuationAmount'
target_correlation = correlation_matrix['PropertyValuationAmount'].sort_values(ascending=False)

target_correlation

# Remove highly correlated features
reduced_features = [f for f in features if f not in ['rolling_mean_2_years']]

# Select new set of features and target variable
X_reduced = df[reduced_features]

# Split the data into training and test sets
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Initialize and train a new linear regression model
lr_model_reduced = LinearRegression()
lr_model_reduced.fit(X_train_reduced, y_train_reduced)

# Predict the target for the test set
y_pred_reduced = lr_model_reduced.predict(X_test_reduced)

# Evaluate the model
mse_reduced = mean_squared_error(y_test_reduced, y_pred_reduced)
rmse_reduced = np.sqrt(mse_reduced)

rmse_reduced


# Lasso Regression
lasso_params = {'alpha': [0.1, 0.5, 1, 5, 10, 20, 50, 100]}
lasso_grid = GridSearchCV(Lasso(random_state=42), lasso_params, cv=5, scoring='neg_root_mean_squared_error')
lasso_grid.fit(X_train, y_train)
best_lasso_model = Lasso(alpha=lasso_grid.best_params_['alpha'], random_state=42)
best_lasso_model.fit(X_train, y_train)
y_pred_best_lasso = best_lasso_model.predict(X_test)
rmse_best_lasso = np.sqrt(mean_squared_error(y_test, y_pred_best_lasso))
print(f"RMSE for Lasso: {rmse_best_lasso}")

# Ridge Regression
ridge_params = {'alpha': [0.1, 0.5, 1, 5, 10, 20, 50, 100]}
ridge_grid = GridSearchCV(Ridge(random_state=42), ridge_params, cv=5, scoring='neg_root_mean_squared_error')
ridge_grid.fit(X_train, y_train)
best_ridge_model = Ridge(alpha=ridge_grid.best_params_['alpha'], random_state=42)
best_ridge_model.fit(X_train, y_train)
y_pred_best_ridge = best_ridge_model.predict(X_test)
rmse_best_ridge = np.sqrt(mean_squared_error(y_test, y_pred_best_ridge))
print(f"RMSE for Ridge: {rmse_best_ridge}")


# Plotting Predictions
plt.figure(figsize=(21, 6))
# Lasso plot
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_best_lasso, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Lasso: True vs Predicted Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
# Ridge plot
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_best_ridge, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Ridge: True vs Predicted Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
# OLS plot
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_reduced, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('OLS: True vs Predicted Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
# Automatically adjust subplot params for better layout
plt.tight_layout()
# Show plot
plt.show()

from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid for Lasso
lasso_params = {'alpha': [0.1, 0.5, 1, 5, 10, 20, 50, 100]}

# Define hyperparameter grid for Ridge
ridge_params = {'alpha': [0.1, 0.5, 1, 5, 10, 20, 50, 100]}

# Initialize GridSearchCV for Lasso
lasso_grid = GridSearchCV(Lasso(random_state=42), lasso_params, cv=5, scoring='neg_root_mean_squared_error')

# Initialize GridSearchCV for Ridge
ridge_grid = GridSearchCV(Ridge(random_state=42), ridge_params, cv=5, scoring='neg_root_mean_squared_error')

# Fit the models
lasso_grid.fit(X_train_reduced, y_train_reduced)
ridge_grid.fit(X_train_reduced, y_train_reduced)

# Get the best parameters and best RMSE score for each model
best_params_lasso = lasso_grid.best_params_
best_rmse_lasso = -lasso_grid.best_score_  # Convert back to positive RMSE

best_params_ridge = ridge_grid.best_params_
best_rmse_ridge = -ridge_grid.best_score_  # Convert back to positive RMSE

best_params_lasso, best_rmse_lasso, best_params_ridge, best_rmse_ridge # {'alpha': 50} 2905249.9344099322 {'alpha': 100} 2905249.935449501

# Use the best parameters to initialize new Lasso and Ridge models
best_lasso_model = Lasso(alpha=best_params_lasso['alpha'], random_state=42)
best_ridge_model = Ridge(alpha=best_params_ridge['alpha'], random_state=42)

# Train the best Lasso model
best_lasso_model.fit(X_train_reduced, y_train_reduced)
y_pred_best_lasso = best_lasso_model.predict(X_test_reduced)
rmse_best_lasso = np.sqrt(mean_squared_error(y_test_reduced, y_pred_best_lasso))

# Train the best Ridge model
best_ridge_model.fit(X_train_reduced, y_train_reduced)
y_pred_best_ridge = best_ridge_model.predict(X_test_reduced)
rmse_best_ridge = np.sqrt(mean_squared_error(y_test_reduced, y_pred_best_ridge))

rmse_best_lasso, rmse_best_ridge # 2671271.2553649493 2671271.106871828

# Get the coefficients from the best-tuned Lasso and Ridge models
lasso_coefficients = pd.Series(best_lasso_model.coef_, index=X_train_reduced.columns).sort_values(ascending=False)
ridge_coefficients = pd.Series(best_ridge_model.coef_, index=X_train_reduced.columns).sort_values(ascending=False)

# Display the coefficients for interpretation
print(lasso_coefficients)
lasso_coefficients, ridge_coefficients

#! Create df

from sklearn.metrics import mean_absolute_error, mean_squared_error

ols = LinearRegression()
ols.fit(X_train_reduced, y_train_reduced)

# Predictions using OLS, Lasso, and Ridge
ols_preds = ols.predict(X_test_reduced)
lasso_preds = best_lasso_model.predict(X_test_reduced)
ridge_preds = best_ridge_model.predict(X_test_reduced)

# Calculate MAE and MSE for each model
metrics_data = {
    'Model': ['OLS', 'Lasso', 'Ridge'],
    'MAE': [
        mean_absolute_error(y_test_reduced, ols_preds),
        mean_absolute_error(y_test_reduced, lasso_preds),
        mean_absolute_error(y_test_reduced, ridge_preds)
    ],
    'MSE': [
        mean_squared_error(y_test_reduced, ols_preds),
        mean_squared_error(y_test_reduced, lasso_preds),
        mean_squared_error(y_test_reduced, ridge_preds)
    ]
}

# Create a DataFrame to display the results
metrics_df = pd.DataFrame(metrics_data)
print(metrics_df)

#! PLOT PREDICTIONS

import matplotlib.pyplot as plt

# Plot true values vs. predicted values for Lasso
plt.figure(figsize=(14, 6))
# Lasso plot
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_reduced, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('OLS: True vs Predicted Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.subplot(1, 3, 2)
plt.scatter(y_test_reduced, y_pred_best_lasso, alpha=0.5)
plt.plot([y_test_reduced.min(), y_test_reduced.max()], [y_test_reduced.min(), y_test_reduced.max()], 'r--')
plt.title('Lasso: True vs Predicted Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
# Plot true values vs. predicted values for Ridge
plt.subplot(1, 3, 3)
plt.scatter(y_test_reduced, y_pred_best_ridge, alpha=0.5)
plt.plot([y_test_reduced.min(), y_test_reduced.max()], [y_test_reduced.min(), y_test_reduced.max()], 'r--')
plt.title('Ridge: True vs Predicted Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.tight_layout()
plt.show()

#! Plot learning curve

from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve."""
    plt.figure(figsize=(8, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt

# Plot learning curve for Lasso
plot_learning_curve(best_lasso_model, "Learning Curve (Lasso)", X_train_reduced, y_train_reduced, cv=5)
plt.show()

# Plot learning curve for Ridge
plot_learning_curve(best_ridge_model, "Learning Curve (Ridge)", X_train_reduced, y_train_reduced, cv=5)
plt.show()

from sklearn.model_selection import validation_curve

def plot_validation_curve(estimator, title, X, y, param_name, param_range, ylim=None, cv=None, n_jobs=None):
    """Generate a simple plot of the test and training validation curve."""
    plt.figure(figsize=(8, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range, cv=cv, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.semilogx(param_range, train_scores_mean, 'o-', color="r", label="Training score")
    plt.semilogx(param_range, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt

# Define the range of alpha values to consider
alpha_range = [0.01, 0.1, 1, 10, 100]

# Plot validation curve for Lasso
plot_validation_curve(Lasso(random_state=42), "Validation Curve (Lasso)", X_train_reduced, y_train_reduced, param_name="alpha", param_range=alpha_range, cv=5)
plt.show()

# Plot validation curve for Ridge
plot_validation_curve(Ridge(random_state=42), "Validation Curve (Ridge)", X_train_reduced, y_train_reduced, param_name="alpha", param_range=alpha_range, cv=5)
plt.show()
