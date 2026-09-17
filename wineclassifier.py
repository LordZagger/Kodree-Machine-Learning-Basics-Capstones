#Wine Classifier - Kodree (Capstone #1)
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#Step 1: Load and prepare
wine = load_wine()
X = wine.data
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) #only fit on training data
X_test_scaled = scaler.transform(X_test) #scale test data based on training data fitting

#Step 2: Train 3 classifiers
DTC = DecisionTreeClassifier(random_state=42) #decision tree classifier
KNC = KNeighborsClassifier(n_neighbors=5) #K Neighbors classifier
LR = LogisticRegression(max_iter = 1000, random_state=42) #Logistic Regression

DTC.fit(X_train_scaled, y_train)
DTC_preds = DTC.predict(X_test_scaled)
DTC_accuracy = accuracy_score(y_test, DTC_preds)

KNC.fit(X_train_scaled, y_train)
KNC_preds = KNC.predict(X_test_scaled)
KNC_accuracy = accuracy_score(y_test, KNC_preds)

LR.fit(X_train_scaled, y_train)
LR_preds = LR.predict(X_test_scaled)
LR_accuracy = accuracy_score(y_test, LR_preds)

#Step 3: Measure
print(f"DTC accuracy score: {round(DTC_accuracy,3)}")
print(f"KNC accuracy score: {round(KNC_accuracy,3)}")
print(f"LR accuracy score: {round(LR_accuracy,3)}")

#Step 4: Visualize
plt.figure()
names = ["DTC", "KNC", "LR"]
scores = [DTC_accuracy, KNC_accuracy, LR_accuracy]
plt.bar(names, scores)
plt.title("Visual comparison of models' accuracy scores")
plt.ylabel("Accuracy score")
plt.ylim(0,1.1)
plt.show()

#Step 5: Investigate
def identify_best_model(scores_list):
    """
    uses the accuracy scores to determine which model performed the best
    returns the index so a later if-else can proceed with the right confusion matrix
    """
    best_score = max(scores_list)
    best_score_index = scores_list.index(best_score)
    return best_score_index

plt.figure()
max_score_index = identify_best_model(scores)
if max_score_index == 0:
    DTC_cm = confusion_matrix(y_test, DTC_preds)
    print(DTC_cm)
    
    ConfusionMatrixDisplay.from_predictions(y_test, DTC_preds)
    plt.title("DTC Confusion Matrix")
    plt.show()
elif max_score_index == 1:
    KNC_cm = confusion_matrix(y_test, KNC_preds)
    print(KNC_cm)
    
    ConfusionMatrixDisplay.from_predictions(y_test, KNC_preds)
    plt.title("KNC Confusion Matrix")
    plt.show()
else: #has to be 2
    LR_cm = confusion_matrix(y_test, LR_preds)
    print(LR_cm)
    
    ConfusionMatrixDisplay.from_predictions(y_test, LR_preds)
    plt.title("LR Confusion Matrix")
    plt.show()

#Step 6: Conclusions
#Q1: The Logistic Regression performed the best, with an accuracy of 1.0 (100%)

#Q2: The class of the confusion matrices' second-row had the most confusion overall (2), but particularly with the KNeighbors classifier
#KNeighbors confusion matrix below to see what I mean:
#    [[14  0  0]
#     [ 1 12  1]   <<<<<-------
#     [ 0  0  8]]

#Q3: Based on the scatter plot provided in the instructions, it's clear that the classes overlap at some of their boundaries;
#data points on the edges of the green class, where the blue or red data points begin their edges (start to crowd), could've led KNeighbors classification to view one of those other classes as the better fit
