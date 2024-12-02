# Breast Cancer Classification with Multiple Machine Learning Models  

This repository contains a Python script that trains and evaluates several machine learning models to classify breast cancer using the **Breast Cancer Wisconsin Dataset** from **scikit-learn**. The results, including model accuracy, are saved to a text file (`metrics.txt`).

---

## **Features**  
- Models used:  
  - Random Forest  
  - Logistic Regression  
  - K-Nearest Neighbors  
  - Decision Tree  
- Data scaling with **StandardScaler**.  
- Accuracy results are logged in a `metrics.txt` file.  

---

## **Requirements**  

Install the required packages using the following command:  

```bash  
pip install numpy pandas scikit-learn
----------------
## ** Usage

1. Run the Script
  To run the script and evaluate the models:

python main.py  
2. Output
  The script prints the accuracy of each model in the console.
  Results are saved in a file named metrics.txt in the following format:
Model: Random Forest  
Accuracy: 0.9649  
----------------------------------------  
Model: Logistic Regression  
Accuracy: 0.9737  
----------------------------------------  
## Script Overview

Data Loading:
  * Loads the Breast Cancer Wisconsin dataset using load_breast_cancer from scikit-learn.
Data Preprocessing:
  * Splits the data into training and testing sets using train_test_split.
  * Scales the data using StandardScaler to normalize features.
Model Training and Evaluation:
  * Trains the following models:
    - Random Forest
    - Logistic Regression
    - K-Nearest Neighbors
    - Decision Tree
  * Calculates and logs model accuracy using accuracy_score.

## Model Configuration

You can modify the models dictionary in the script to add or update models and their hyperparameters:

models = {  
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),  
    "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=200),  
    "K-Nearest Neighbors": KNeighborsClassifier(),  
    "Decision Tree": DecisionTreeClassifier(random_state=random_state)  
}  
Contributions

Contributions are welcome! Feel free to fork the repository, create a branch, and submit a pull request.

