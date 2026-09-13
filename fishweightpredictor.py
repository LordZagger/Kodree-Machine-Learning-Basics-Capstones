import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

#Step 1: Load and Explore
df = pd.read_csv("C:/Users/bijou/Documents/waterloo pdfs/CoopSpring2025+Fall2025/Extra/fish.csv")
print(df.shape)
print(df.columns)
print(df.head())
print(df.describe())

#Step 2: Prepare features
X = df.drop(columns=["Species", "Weight"]).values
y = df["Weight"].values
print(X.shape)
print(y.shape)

#Step 3: Visualization
plt.figure()
plt.scatter(df["Length1"].values, y)
plt.title("Weight (y) vs Length1 (x) for fish dataset")
plt.xlabel("Length1")
plt.ylabel("Weight")
plt.show()
#scatter plot clearly shows a somewhat exponential relationship, therefore non-linear regression is better

#Step 4: Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Step 5: Linear baseline
LR = LinearRegression()
LR.fit(X_train, y_train)
LR_preds = LR.predict(X_test)
LR_R2 = r2_score(y_test, LR_preds)
LR_MSE = mean_squared_error(y_test, LR_preds)
print(f"R²: {LR_R2:.3f}")
print(f"MSE: {LR_MSE:.3f}")

#Step 6: Unlimited tree
DTR1 = DecisionTreeRegressor(random_state=42)
DTR1.fit(X_train, y_train)
DTR1_r2_train = r2_score(y_train, DTR1.predict(X_train))
DTR1_r2_test = r2_score(y_test, DTR1.predict(X_test))
print(f"Decision Tree (default depth) Train R²: {DTR1_r2_train:.3f}")
print(f"Decision Tree (default depth) Test R²: {DTR1_r2_test:.3f}")
gap = DTR1_r2_train - DTR1_r2_test
print(f"R² gap: {gap:.3f}")
#With a train R² of 1 and a test R² of 0.939, the model does not seem to be overfitting (small gap of 0.061 between R²s)

#Step 7: Find the best depth
depths_list = [2,3,4,5,6,8,10]
r2_scores = []
for depth in depths_list:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    print(f"Depth={depth}: R²={r2:.3f}")
    r2_scores.append(r2)

max_r2 = max(r2_scores)
max_r2_index = r2_scores.index(max_r2)
best_depth = depths_list[max_r2_index]
print(f"Best depth: {best_depth}")

#Step 8: Compare all three
print(f"Linear Regression Test R²: {LR_R2:.3f}")
print(f"Decision Tree (default depth) Test R²: {DTR1_r2_test:.3f}")
print(f"Decision Tree (best depth) Test R²: {max_r2:.3f}")

plt.figure()
names = ["Linear Regression", "Decision Tree (default depth)", "Decision Tree (best depth)"]
scores = [LR_R2, DTR1_r2_test, max_r2]
plt.bar(names, scores)
plt.title("Test R² scores for different models trained on fish market dataset")
plt.ylabel("R² score")
plt.ylim(0,1.1)
plt.show()

#Step 9: Evaluate the best tree
best_model = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
best_model.fit(X_train, y_train)
best_predictions = best_model.predict(X_test)
residuals = y_test - best_predictions

plt.figure()
plt.subplot(1,2,1)
plt.scatter(y_test, best_predictions)
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],color="red",linestyle="--")
plt.xlabel("Actual weights")
plt.ylabel("Predicted weights")
plt.title("Actual vs predicted weights scatter plot for best DTR model")

plt.subplot(1,2,2)
plt.scatter(best_predictions, residuals)
plt.axhline(y=0, color="red",linestyle="--")
plt.title("Residual plot for best DTR model")
plt.xlabel("Predicted weights")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()

#Step 10: Conclusions
#Q1: Linear Regression's R² was 0.882, and did not do as well the Decision Tree Regressors because the data points had a more exponential trend than linear
    
#Q2: The best max_depth was 5, and its test R² was the highest of all the depths at 0.958 (better than the default depth 0.939)
    
#Q3: The unlimited tree scored perfectly on the training set (train R²=1.0), and very good on the test set (train R²=0.939); the small gap between R² indicates that the model did not overfit
        
#Q5: There are no evident patterns based on the residuals plot; there are some overestimations and some underestimations across the entire predicted weights range, with the only 2 outliers being the more than 300 residual at the 1000 predicted weight and the near 200 residual at 350-400 predicted weight
