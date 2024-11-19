# import libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score


# read dataset
data = load_breast_cancer()
x = data.data
y = data.target


# define variables
random_state = 12
test_size = 0.15

# split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

#define models dictionary
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
    "Logistic Regression": LogisticRegression(random_state=random_state),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=random_state)
}
   
#function to train & evaluate each model --> save accuracy  
def train_and_evaluate_model(model_name, model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred)   

    # save model accuracy in file
    with open("metrices.txt", "a") as file: # "a" to append to the file
        file.write(f"Model: {model_name}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write("-"*40 + "\n")

for model_name, model in models.items():
    train_and_evaluate_model(model_name, model, x_train, x_test, y_train, y_test)
   
