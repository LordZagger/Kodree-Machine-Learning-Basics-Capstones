#California Housing Predictor (Linear regression model) - Kodree (Capstone #2)
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

#Step 1: Load and Explore
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
print(df.head())
print(df.tail())
print(df.shape)
print(housing.feature_names)

#Step 2: Split
X = housing.data
y = housing.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Step 3: Scale
#I tried scaling, and it does not affect mse and r2 much with this dataset and linear regression model, so no real point in scaling

#Step 4:Train
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

#Step 5: Measure
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"MSE: {round(mse,2)}")
print(f"R²: {round(r2,3)}")
print(f"The model explains {round(r2,3)*100}% of the price variation.")

#Step 6: Predicted vs actual plot
plt.figure()
plt.scatter(y_test, predictions)
plt.title("California Housing - Actual vs predicted comparison")
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],color="red",linestyle="--")
plt.show()

#Step 7: Residual plot
residuals = y_test - predictions
plt.figure()
plt.scatter(predictions, residuals)
plt.axhline(y=0,color="red",linestyle="--")
plt.title("California Housing - Residuals plot")
plt.xlabel("Predicted price")
plt.ylabel("Residuals")
plt.show()

#Step 8: Conclusions
#Q1: R² is 0.576, which tells me the model explains 57.6% of the price variation (which isn't actually that great)
    
#Q2: Based on the scatter plot, the model is most accurate for the 0 to 2.5 actual price range, as there is an equal balance of under and overpredictions in this actual price range
    
#Q3: (Sinisterly), there is a pattern from the residuals: the residuals seem to have a downward trend as the predicted price increases
#The model starts (before 0 of predicted price) by overestimating; then from 0 to 4, there is an equal balance of under and overestimating; then after 4, the model mostly underestimates

#Q4: Based on all these prior conclusions, a more complex model would better fit this housing price predictor
