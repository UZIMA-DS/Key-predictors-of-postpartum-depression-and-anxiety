# run_classifier.py
import pandas as pd
#from logistic_module import Classifier  # import your Classifier class
from rf_xgboost_module import Classifier

# Load your data
X = pd.read_csv(r"C:\Users\faith.neema\Downloads\features.csv")
y = pd.read_csv(r"C:\Users\faith.neema\Downloads\labels.csv")

#if __name__ == "__main__":
    #classifier_logistic = Classifier(classifier='logistic')
    #classifier_logistic.train_evaluate(
    #    10,
     #   X,
     #   y,
     #   scoring=classifier_logistic.scoring,
      #  metric=list(classifier_logistic.scoring.keys())[0]
    #)

#if __name__ == "__main__":
    #classifier_rf = Classifier(classifier='rf')
    #classifier_rf.train_evaluate(
        #10,
        #X,
        #y,
        #scoring=classifier_rf.scoring,
        #metric=list(classifier_rf.scoring.keys())[0]
    #)

if __name__ == "__main__":
    classifier_xgb = Classifier(classifier='xgb')
    classifier_xgb.train_evaluate(
      10,
        X,
       y,
       scoring=classifier_xgb.scoring,
       metric=list(classifier_xgb.scoring.keys())[0])