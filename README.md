### Project Title: Key predictors of postpartum depression and anxiety symptoms among mothers in Kilifi, Kenya: a machine learning approach

**Description**

This project trains and evaluates machine learning models (Random Forest, XGBoost, Logistic Regression) to predict postpartum depression and anxiety symptoms.
It also incorporates model explainability using SHAP analysis.

-Postpartum depression and anxiety data cleaning and preparation.ipynb - cotains data cleaning and preparation process.
- logistic_module.py - contains the Classifier class (for logistic regression or RF).
- rf_xgboost_module.py - contains the Classifier class for Random Forest + XGBoost models with SHAP explainability.
- run_classifier.py - Main script to load data, train models, and evaluate.

**Installation**

***1. Clone repository***

git clone [your-repo-url]

cd project_folder

***2. Create a virtual environment***

python -m venv env

***3. Install dependencies***

pip install -r requirements.txt

**Usage**

1. Prepare your data

  Make sure your data files are in the project folder:
  - features.csv - feature columns
  - anxiety_labels.csv -target labels for anxiety
  - depression_labels.csv -target labels for depression

2. Run the main script

  py -3 run_classifier.py

  - This will train the model(s) and evaluate them using cross-validation.
  - Metrics such as accuracy, precision, recall, F1-score, ROC-AUC, and brier score will be printed.

3. SHAP Visualizations (optional)

   - SHAP summary and dependence plots are generated to explain feature importance.
   - If you don’t need SHAP plots, you can comment out the lines in rf_xgboost_module.py causing plotting issues.
  
**Dependencies**

From requirements.txt 

pandas

numpy

scikit-learn

matplotlib

seaborn

shap

imbalanced-learn

statsmodels

pingouin

tableone

missingno

xgboost

joblib

scipy

**Running the Classifier (run_classifier.py)**

The run_classifier.py script is the main entry point for training and evaluating the machine learning models. It supports Random Forest, XGBoost, and Logistic Regression (though only one model is run at a time).

***How it works***
1.	Imports

- pandas is used to load the dataset (features.csv and labels.csv).
- Classifier is imported from your module (rf_xgboost_module.py) to handle model training, evaluation, and optional SHAP explainability.

2.	Load data

X = pd.read_csv("features.csv")  # feature columns

y = pd.read_csv("anxiety_labels.csv")    # target labels, this can be replaced with depression_labels

- X - input features

- y - target labels

3 .	Run a model

if __name__ == "__main__":
    classifier_rf = Classifier(classifier='rf')
    classifier_rf.train_evaluate(
        10,
        X,
        y,
        scoring=classifier_rf.scoring,
        metric=list(classifier_rf.scoring.keys())[0]
    )

- Classifier(classifier='rf') - initializes a Random Forest classifier.
- train_evaluate() - trains the model using 10-fold cross-validation.
- scoring - dictionary of metrics (accuracy, precision, recall, F1-score, ROC-AUC).
- metric - specifies the main metric for model selection.

4.	Optional models

- 	Logistic Regression:

classifier_logistic = Classifier(classifier='logistic')

classifier_logistic.train_evaluate(...)

- XGBoost:

classifier_xgb = Classifier(classifier='xgb')

classifier_xgb.train_evaluate(...)

You can uncomment these lines to train and evaluate other classifiers.

5.	Outputs
-	Model performance metrics printed in the console.
-	Optionally, SHAP plots can be generated for model explainability

6. Hyperparameters used:

- For anxiety use:

  # Hyperparameters for RandomForestClassifier

  n_estimators =[100] 

  max_depth =[3]

  min_samples_leaf =[28]

  max_features =[9]

  # Hyperparameters for xgboostClassifier

  n_estimators=[3000]

  max_depth = [1]

  learning_rate = [0.01]

scale_pos_weight=10.565

  - For depression use:

  # Hyperparameters for RandomForestClassifier

  n_estimators =[200] 

  max_depth =[3]

  min_samples_leaf =[24]

  max_features =[14]

  # Hyperparameters for xgboostClassifier

  n_estimators=[1000]

  max_depth = [1]

  learning_rate = [0.01]

scale_pos_weight=5.595




