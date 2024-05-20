
################################################################ Packages ################################################################
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge


################################################################ Loading Data ################################################################
dataset_path = "./DataPreparation/merged_dataset_clean.csv"
data = pd.read_csv(dataset_path, delimiter=';')



################################################################ Preprocess Data ################################################################
# checking for missing values in the dataset
missing_values = data.isnull().sum()

# checking the data types of the columns
data_types = data.dtypes

print(missing_values, data_types)


# we need to insure there are no missing values and that every column is of the correct type

# converting MunicipalitySize and PropertyTaxRate columns to float by replacing commas
data['MunicipalitySize'] = data['MunicipalitySize'].str.replace(',', '').astype(float)
data['PropertyTaxRate'] = data['PropertyTaxRate'].str.replace(',', '').astype(float)
data['MunicipalityUnemployment'] = data['MunicipalityUnemployment'].str.replace(',', '').astype(float)

# filling the missing values in MunicipalityAverageIncome with the mean of observations for the same Municipality
data['MunicipalityAverageIncome'] = data.groupby('Municipality')['MunicipalityAverageIncome'].transform(lambda x: x.fillna(x.mean()))

# confirming that no missing values remain
missing_values_updated = data.isnull().sum()
missing_values_updated



# convert the ValuationChangeDate column to a datetime object
data['ValuationChangeDate'] = pd.to_datetime(data['ValuationChangeDate'])

# extract year, month, quarter, and day of the week from ValuationChangeDate
data['Year_Valuation'] = data['ValuationChangeDate'].dt.year
data['Month_Valuation'] = data['ValuationChangeDate'].dt.month
data['Quarter_Valuation'] = data['ValuationChangeDate'].dt.quarter


# one-hot encode the Municipality column
data_one_hot = pd.get_dummies(data, columns=['Municipality'], drop_first=True)

# drop the address and original ValuationChangeDate columns
data_one_hot.drop(columns=['address', 'ValuationChangeDate'], inplace=True)

# split the processed data into features and target variable
X_processed = data_one_hot.drop(columns=['PropertyValuationAmount'])
y_processed = data_one_hot['PropertyValuationAmount']

# split the processed data into two sets; 80% training & 20% testing
X_train_processed, X_test_processed, y_train_processed, y_test_processed = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)

# Standard scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)



################################################################ Machine Learning #################################################################
# OLS:
# fitting a linear regression model to the preprocessed training data
ols_linear_model = LinearRegression()
ols_linear_model.fit(X_train_processed, y_train_processed)

# predicting on the test set
ols_y_pred= ols_linear_model.predict(X_test_processed)

# calculating the Mean Squared Error (MSE) for the Linear Regression model
ols_mse = mean_squared_error(y_test_processed, ols_y_pred)
ols_mae = mean_absolute_error(y_test_processed, ols_y_pred)
print("OLS MSE: ", ols_mse, "& OLS MAE: ", ols_mae)


# Lasso:
# Defining a range of lambda values for Lasso regularization
lambdas_lasso = np.logspace(-5, 5, 20)

# Performing 5-fold cross-validation to find the optimal lambda for Lasso
lasso = LassoCV(alphas=lambdas_lasso, cv=5, random_state=161193)
lasso.fit(X_train_processed, y_train_processed) # Using the preprocessed data

# Optimal lambda value from cross-validation
optimal_lambda_lasso = lasso.alpha_
print("optimal lambda:", optimal_lambda_lasso)

# Predicting on the test set (no need to refit)
y_pred_lasso = lasso.predict(X_test_processed)

# Calculating the Mean Squared Error (MSE) for the Lasso Regression model
mse_lasso = mean_squared_error(y_test_processed, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test_processed, y_pred_lasso)
print("Lasso MSE:", mse_lasso, "& Lasso MAE:", mae_lasso)



# Ridge:
# Defining a range of lambda values for Ridge regularization
lambdas_ridge = np.logspace(-5, 5, 20)

# Performing cross-validation to find the optimal lambda for Ridge with scaled features
ridge = RidgeCV(alphas=lambdas_ridge)
ridge.fit(X_train_scaled, y_train_processed) # Using the scaled data

# Optimal lambda value from cross-validation with scaled features
optimal_lambda_ridge = ridge.alpha_
print("optimal lambda:", optimal_lambda_ridge)

# Predicting on the scaled test set
y_pred_ridge = ridge.predict(X_test_scaled)

# Calculating the Mean Squared Error (MSE) for the Ridge Regression model
mse_ridge = mean_squared_error(y_test_processed, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test_processed, y_pred_ridge)
print("Ridge MSE:", mse_ridge, "& Ridge MAE:", mae_ridge)


# Creating a DataFrame with the MSEs and MAEs
results_df = pd.DataFrame({
    'Model': ['OLS Regression', 'Lasso Regression', 'Ridge Regression'],
    'MSE': [ols_mse, mse_lasso, mse_ridge],
    'MAE': [ols_mae, mae_lasso, mae_ridge]
})

# Displaying the DataFrame
print(results_df)


################################################################ Residuals #################################################################

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

def plot_predictions(y_true, y_pred, model_name, ax):
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'Predicted vs Actual Values for {model_name}')

# OLS Predictions Plot
plot_predictions(y_test_processed, ols_y_pred, 'OLS Regression', axes[0])

# Lasso Predictions Plot
plot_predictions(y_test_processed, y_pred_lasso, 'Lasso Regression', axes[1])

# Ridge Predictions Plot
plot_predictions(y_test_processed, y_pred_ridge, 'Ridge Regression', axes[2])

plt.show()



################################################################ Learning Curve #################################################################
def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='neg_mean_squared_error'
    )
    
    # Mean of cross-validation scores across the folds
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(7,3))
    
    # Fill the area for the test curve
    ax.fill_between(train_sizes, -test_scores.min(1), -test_scores.max(1), alpha=0.25, label='Test', color='blue')
    
    # Plot the training curve as a simple line
    ax.plot(train_sizes, train_scores_mean, 'o-', label='Train', color='orange')
    
    ax.set_title(title)
    ax.set_xlabel('Training set size')
    ax.set_ylabel('Mean Squared Error')
    ax.legend()
    plt.show()


# For OLS Regression
#plot_learning_curve(ols_linear_model, X_train_processed, y_train_processed, 'Learning Curve for OLS Regression')


# For Lasso Regression
#plot_learning_curve(lasso, X_train_processed, y_train_processed, 'Learning Curve for Lasso Regression')

# For Ridge Regression
plot_learning_curve(ridge, X_train_scaled, y_train_processed, 'Learning Curve for Ridge Regression')



################################################################ Validation Curve #################################################################
def plot_validation_curve(model, X, y, param_name, param_range, optimal_value, title):
    train_scores, test_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range, cv=5, scoring="neg_mean_squared_error", n_jobs=1)
    
    # Take the mean of cross-validation scores across the folds
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the curves without dots ('-' instead of 'o-')
    ax.plot(param_range, test_scores_mean, '-', label='Test', color='blue')
    ax.plot(param_range, train_scores_mean, '-', label='Train', color='orange')

    # Adding a vertical dotted line at the optimal hyperparameter value
    ax.axvline(x=optimal_value, color='red', linestyle='--', label='Optimal Value')

    ax.set_title(title)
    ax.set_xlabel('Hyperparameter Value (log scale)')
    ax.set_ylabel('Mean Squared Error')
    ax.set_xscale('log')
    ax.legend()
    plt.show()
    
# Optimal value for Lasso Regression
#optimal_lambda_lasso = lasso.alpha_
#plot_validation_curve(Lasso(), X_train_processed, y_train_processed, param_name="alpha", param_range=lambdas_lasso, optimal_value=optimal_lambda_lasso, title="Validation Curve for Lasso Regression")

# Optimal value for Ridge Regression
optimal_lambda_ridge = ridge.alpha_
plot_validation_curve(Ridge(), X_train_scaled, y_train_processed, param_name="alpha", param_range=lambdas_ridge, optimal_value=optimal_lambda_ridge, title="Validation Curve for Ridge Regression")

   