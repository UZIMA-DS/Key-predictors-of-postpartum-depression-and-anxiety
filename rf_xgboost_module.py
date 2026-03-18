import pandas as pd
import numpy as np
import seaborn as sns
import shap
from matplotlib.backends.backend_pdf import PdfPages
import os
from joblib import dump
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import make_scorer, roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from scipy.stats import chi2
from statsmodels.stats.proportion import proportion_confint


import warnings
    # Ignore DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ==============================
# CLASSIFIER CLASS
# ==============================


class Classifier(BaseEstimator, TransformerMixin):
    def __init__(self, classifier='rf'):
      # Store classifier type
        self.classifier = classifier
        
        # Hyperparameters for RandomForestClassifier
        self.rf_n_estimators =[1500] # 1500
        self.rf_max_depth =[6]# 6
        self.rf_min_samples_split = [10]#10
        self.rf_min_samples_leaf =[26]# 26The minimum number of samples required to be in a leaf node (the end of a branch in a tree).Higher values prevent overfitting by ensuring leaf nodes are not too small.
        self.rf_max_features =[16]#16 (The number of features to consider when splitting a node.Larger values may improve accuracy if more features are useful.)
        self.rf_bootstrap = [True]

        # Hyperparameters for XGBoostClassifier
        self.xgb_n_estimators=[600]#300, 600, 900
        self.xgb_max_depth = [3]#2,4,6
        self.xgb_learning_rate = [0.0001]#0.0001,0.001,0.01,0.1
        self.xgb_min_child_weight = [0.5]#0.7
        self.xgb_colsample_bytree = [0.5]#0.7
        self.xgb_reg_alpha = [0.8]#0.4
        self.xgb_reg_lambda = [0.8]#0.4
        
        
    
        

        # Hyperparameters for LightGBMClassifier
        self.lgbm_n_estimators = [200, 400, 800]
        self.lgbm_max_depth = [2, 3, 4]
        self.lgbm_learning_rate = [0.5, 0.1, 0.01, 0.001]
        self.lgbm_num_leaves = list(np.arange(start=20, stop=100, step=20))

        self.lgbm_min_data_in_leaf = list(np.arange(start=200, stop=1000, step=100))

        self.lgbm_max_bin = [200, 300]
        self.lgbm_lambda_l1 = list(np.arange(start=0, stop=50, step=5))

        self.lgbm_lambda_l2 = list(np.arange(start=0, stop=50, step=5))

        self.lgbm_bagging_fraction = list(np.arange(start=0.2, stop=0.95, step=0.1))

        self.lgbm_bagging_freq = [1]
        self.lgbm_feature_fraction = list(np.arange(start=0.2, stop=0.92, step=0.1))

        # Classifier pipeline
        self.pipe = Pipeline(steps=[ ('classifier', self.create_classifier())])

        #  # Scoring metrics used in GridSearchCV
        self.scoring = {'auc': 'roc_auc', 
                        'accuracy': make_scorer(accuracy_score),
                        'precision': make_scorer(precision_score, average='macro'),
                        'recall': make_scorer(recall_score, average='macro'),
                        'f1': make_scorer(f1_score, average='macro'), 
                        'average_precision':make_scorer(average_precision_score)}
   # -----------------------------
    # Create models
    # -----------------------------
    
    def create_classifier(self):
        if self.classifier == 'rf':
            return RandomForestClassifier(random_state=42, class_weight='balanced')#class_weight='balanced'
        elif self.classifier == 'xgb':
            
            return xgb.XGBClassifier(tree_method='auto', random_state=42,scale_pos_weight=8.61)  
        elif self.classifier == 'lgbm':
            return lgb.LGBMClassifier(random_state=42, is_unbalance = True)
        else:
            raise ValueError("Invalid classifier option")
    # -----------------------------
    # Hyperparameter grid
    # -----------------------------  
    def get_grid(self):
        # Define hyperparameter grid for GridSearch
        if self.classifier == 'rf':
            return {'classifier__n_estimators': self.rf_n_estimators,#number of trees in the ensemble
            'classifier__max_depth': self.rf_max_depth, #depth of each tree, smaller values prevent overfitting by limiting the growth of trees (shallow trees)
              'classifier__min_samples_split': self.rf_min_samples_split,
                   'classifier__max_features':self.rf_max_features,
                   'classifier__min_samples_leaf':self.rf_min_samples_leaf,
                'classifier__bootstrap':self.rf_bootstrap}
                   
        elif self.classifier == 'xgb':
            return {'classifier__n_estimators': self.xgb_n_estimators,
                    'classifier__max_depth': self.xgb_max_depth,
                    'classifier__learning_rate': self.xgb_learning_rate,
                'classifier__min_child_weight': self.xgb_min_child_weight,
                'classifier__colsample_bytree': self.xgb_colsample_bytree}
                #'classifier__reg_alpha': self.xgb_reg_alpha,
                #  'classifier__reg_lambda': self.xgb_reg_lambda }
        elif self.classifier == 'lgbm':
            return {'classifier__n_estimators': self.lgbm_n_estimators,
                    'classifier__max_depth': self.lgbm_max_depth,
                    'classifier__learning_rate': self.lgbm_learning_rate,
                    'classifier__num_leaves': self.lgbm_num_leaves,
                    'classifier__min_data_in_leaf': self.lgbm_min_data_in_leaf,
                    'classifier__max_bin': self.lgbm_max_bin,
                   'classifier__lambda_l1': self.lgbm_lambda_l1,
                   'classifier__lambda_l2': self.lgbm_lambda_l2,
                   'classifier__bagging_fraction': self.lgbm_bagging_fraction,
                   'classifier__bagging_freq': self.lgbm_bagging_freq,
                   'classifier__feature_fraction': self.lgbm_feature_fraction}
        else:
            raise ValueError("Invalid classifier option")
    # -----------------------------
    # Train-test split
    # -----------------------------
    def split_train_test(self, df, y):
        # Split dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=42, test_size=0.2, stratify=y)
        
        #Calculate ratio of negative and positives sample to be used for class balancing with XGBoost
        neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
        print( neg_pos_ratio)
        
        # Count class distribution
        num_positive_samples = sum(y_train == 1)  # Number of positive samples
        num_negative_samples = sum(y_train == 0)  # Number of negative samples
        num_positive_sampless = sum(y_test == 1)
       
        # Print baseline AUPRC
        print(f'Baseline train auprcs: { num_positive_samples/y_train.shape[0]}')
        print(f'Baseline test auprcs: { num_positive_sampless/y_test.shape[0]}')
        
        #----this part to be applying if using smote class imbalanced handling
        #smote = SMOTE(sampling_strategy='auto', random_state=42)
        #X_train, y_train = smote.fit_resample(X_train, y_train)
        return X_train, X_test, y_train, y_test
    
    # -----------------------------
    # GridSearch training
    # -----------------------------
    def grid_train(self, pipe, grid, cv, train_features, train_labels, scoring, metric):
        # Stratified cross-validation
        stratified_cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Repeatedstratified cross-validation
        #Repeatedstratified_cv  = RepeatedStratifiedKFold(n_splits=cv, n_repeats=5, random_state=42)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(estimator=pipe, param_grid=grid, cv= stratified_cv, verbose=2, n_jobs=-1, scoring=scoring, refit=metric, return_train_score=True)
        
        # Train model
        grid_search.fit(train_features, train_labels)
        
        # Retrieve best estimator
        best_grid=grid_search.best_estimator_['classifier']
        
        # Calibrate predicted probabilities
        calibrated_model = CalibratedClassifierCV(estimator=best_grid, method='sigmoid',
        cv=10)
        calibrated_model.fit(train_features, train_labels)
        
        print("Best AUC:", grid_search.best_score_)
        print("Best Hyperparameters:", grid_search.best_params_)
        
        return best_grid, calibrated_model
    
    # -----------------------------
    # Prediction
    # -----------------------------
    def grid_test(self, model, test_features):
         # Predict class probabilities
        y_pred = model.predict(test_features)
        return y_pred
    
    
    def grid_test_prob(self, model, test_features):
        # Compute ROC curve
        y_prob = model.predict_proba(test_features)
        # Find optimal threshold maximizing TPR-FPR -younde index
        return y_prob
    
    
   # -----------------------------
    # Optimal Threshold
    # -----------------------------  
    def Find_Optimal_Cutoff2(self, y_test, y_score):
        # Get probability scores
        precision, recall, thresholds = precision_recall_curve(y_test, y_score[:,1])

        # Find the best threshold that maximizes F1-score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
        best_threshold = thresholds[np.argmax(f1_scores)]

        print(f"Best threshold: {best_threshold:.3f}")
        return  best_threshold
    
    # -----------------------------
    # ROC + PR curves
    # -----------------------------
    def roc_auc_binary(self, y_test, y_score, class_label):
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
        
        # Compute AUC
        roc_auc = auc(fpr, tpr)
        
        # Save Model 1 FPR and TPR
        df = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'roc_auc':round(roc_auc,3)})
        df.to_csv('ecd_rf_roc_curve_updated.csv', index=False)
        
        # Compute Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_score[:, 1])
        
        # Compute AUPRC
        auprc = auc(recall, precision)
        n_bootstraps = 1000
        rng_seed = 42  # control reproducibility
       
        bootstrapped_scores = []
        rng = np.random.RandomState(rng_seed)
        for i in range(n_bootstraps):
           # sample indices with replacement
            indices = rng.randint(0, len(y_test ), len( y_test ))
            
           # make sure both classes are present
            if len(np.unique(y_test[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue
            
            # compute ROC AUC for this sample
            score = roc_auc_score(y_test[indices],  y_score[:, 1][indices])
            bootstrapped_scores.append(score)
        
        # convert to numpy array  
        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()
        
        # Computing the lower and upper bound of the 90% confidence interval
        # You can change the bounds percentiles to 0.025 and 0.975 to get
        # a 95% confidence interval instead.
        confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
        #print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format( confidence_lower, confidence_upper))
        
        # Plot ROC curve)
        plt.figure(figsize=(8, 6))
        # plt.plot(fpr, tpr, color='red', linestyle='-')
        plt.plot(fpr, tpr, color='red', linestyle='-', label=f'ROC curve (AUC = {roc_auc:.3f})\n95% CI: [{confidence_lower:.3f} - {confidence_upper:.3f}]')

        plt.text(0.5, 0.2, f'95% CI: [{ confidence_lower:.3f}, {confidence_upper:.3f}]', color='red', fontsize=12, ha='center')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=18)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=18)
        plt.title('Receiver Operating Characteristic Curve', fontsize=18)
        plt.legend(loc='lower right')
        plt.show()
        
        # Plot Precision-Recall curve
        plt.plot(recall, precision, marker='.', label=f'AUPRC = {auprc:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()
        return  roc_auc
        
     # -------------------------------
     #Plot Histograms of Predicted Probabilities
    # -------------------------------   
    def histogram_plots(self, labels, y_prob):
        """
        Plots histograms of predicted probabilities for each class.

        Parameters:
        labels : list or array
            Class labels
        y_prob : ndarray
            Predicted probabilities with shape (n_samples, n_classes)
        """
        plt.figure(figsize=(15, 7))
        for i, v in enumerate(list(labels)):
            ax = plt.subplot(2, 2, i + 1)
            ax.hist(y_prob[:, i], color='blue', edgecolor='black')
            ax.set_title(f'Histogram: {v} class')
            ax.set_xlabel('Predicted Probabilities')
            ax.set_ylabel('Number of values')
        plt.tight_layout()
        plt.show()

    def con_fusin_matrix(self, y_test, y_pred, labels, alpha=0.95):
        """
        Computes and plots confusion matrix with accuracy, sensitivity, 
        and specificity including confidence intervals.

        Parameters:
        y_test : array-like
            True labels
        y_pred : array-like
            Predicted labels
        labels : list
            Class labels
        alpha : float, optional
            Confidence level (default is 0.95)
        """
        cm = confusion_matrix(y_test, y_pred)
        total1=sum(sum(cm))
        
        ## # Accuracy and CI
        accuracy1=(cm[0,0]+cm[1,1])/total1
        acc_ci_lower, acc_ci_upper = proportion_confint(count=(cm[0, 0] + cm[1, 1]), 
                                                        nobs=total1, 
                                                        alpha=alpha, 
                                                        method='wilson')
        #print(f'Accuracy: {accuracy1:.4f} ({acc_ci_lower:.4f} - {acc_ci_upper:.4f})')

        # Sensitivity and CI
        sensitivity1 = cm[1,1]/(cm[1,0]+cm[1,1])
        sens_ci_lower, sens_ci_upper = proportion_confint(count=cm[1, 1], 
                                                          nobs=(cm[1, 0] + cm[1, 1]), 
                                                          alpha=alpha, 
                                                          method='wilson')
        #print(f'Sensitivity: {sensitivity1:.4f} ({sens_ci_lower:.4f} - {sens_ci_upper:.4f})')
        
        # Specificity and CI
        specificity1 = cm[0,0]/(cm[0,0]+cm[0,1])
        spec_ci_lower, spec_ci_upper = proportion_confint(count=cm[0, 0], 
                                                          nobs=(cm[0, 0] + cm[0, 1]), 
                                                          alpha=alpha, 
                                                          method='wilson')
        #print(f'Specificity: {specificity1:.4f} ({spec_ci_lower:.4f} - {spec_ci_upper:.4f})')
        
        # Plot Confusion Matrix
        cm_df = pd.DataFrame(cm, index=labels.tolist(), columns=labels.tolist())
        plt.figure(figsize=(15, 7))
        sns.heatmap(cm_df, annot=True,cmap= "YlOrBr")
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Values')
        plt.xlabel('Predicted Values')
        plt.show()

    
    # -------------------------------
    #Train and Evaluate Model
    # -------------------------------
    def train_evaluate(self, cv, data, y, scoring, metric):
        """
        Trains a model using cross-validation, evaluates performance on training and testing sets,
        plots calibration curves, confusion matrices, histograms, and computes various metrics 
        including ROC-AUC, Brier Score, and Hosmer–Lemeshow test.

        Parameters:
        cv : int or cross-validation generator
            Number of folds or CV splitter
        data : DataFrame or ndarray
            Feature matrix
        y : array-like
            Target labels
        scoring : str or callable
            Metric used for cross-validation
        metric : str
            Metric used for model selection
        """
        # Encode Labels
        label_binarizer = LabelBinarizer().fit(y)
        label_encoder = preprocessing.LabelEncoder()
        y_label = label_encoder.fit_transform(y)
        
        
        # Split Data into Train and Test
        X_train, X_test, y_train, y_test = self.split_train_test(data, y_label)
        
        # Define Grid and Train Model
        grid = self.get_grid()
        best_model, calibration_model = self.grid_train(self.pipe, grid, cv, X_train, y_train, scoring, metric)
       
        # Predict on Training Set
        y_pred_train = self.grid_test(best_model, X_train)
        y_prob_train = self.grid_test_prob(best_model, X_train)
        y_prob_train1 = self.grid_test_prob(calibration_model, X_train)
        
         # Find optimal cutoff
        threshold = self.Find_Optimal_Cutoff2(y_train, y_prob_train1)
        print(threshold)
        
        # Save Trained Model
        dump(best_model, r"C:\Users\faith.neema\OneDrive - Aga Khan University\Documents\Longitudinal studies\logistic_regression_trained_with_21_features.joblib")
        dump(calibration_model,r"C:\Users\faith.neema\OneDrive - Aga Khan University\Documents\Longitudinal studies\logistic_regression_trained_with_21_model.pkl")

        # Predict on Test Set
        y_prob_test = self.grid_test_prob(best_model, X_test)
        y_prob_test1 = self.grid_test_prob(calibration_model, X_test)
        #y_pred_test = self.grid_test(best_model, X_test)
        y_pred_test = np.where( y_prob_test1[:,1] > threshold, 1, 0)
        
        # Save predictions
        pd.DataFrame({'y_pred': y_pred_test, 'y_test': y_test}).to_csv('dep_lr_pred.csv', index=False)
       
        
        # Classification Reports
        print('---Classification Report for Training Data ---')
        print(classification_report(y_train, y_pred_train, zero_division=0, target_names=list(label_binarizer.classes_)))
        print('---Classification Report for Testing Data ---')
        print(classification_report(y_test, y_pred_test, zero_division=0, target_names=list(label_binarizer.classes_)))
        
        #ROC-AUC Curves
        print('---AUC_ROC for Train data ---')
        self.roc_auc_binary(y_train, y_prob_train, label_binarizer.classes_)
        print('Before calibration')
        self.roc_auc_binary(y_test, y_prob_test, label_binarizer.classes_)
        
        # Confusion Matrices
        print('---Confusion matrix for Training Data ---')
        self.con_fusin_matrix(y_train, y_pred_train, label_binarizer.classes_)
        print('---Confusion matrix for Testing Data ---')
        self.con_fusin_matrix(y_test, y_pred_test, label_binarizer.classes_)
        
        # Histograms of Predicted Probabilities
        self.histogram_plots(label_binarizer.classes_, y_prob_train)
        self.histogram_plots(label_binarizer.classes_, y_prob_test)
        
        #Calibration
        y_prob_test_cal1 = calibration_model.predict_proba(X_test)[:, 1]
        ##y_prob_test_cal = calibrated_model.predict_proba(X_calib)[:, 1]
        
        # Calculate Brier score
        brier_score = brier_score_loss(y_test,   y_prob_test[:, 1])
        print(f'Brier Score before calibration: {brier_score}')
        brier_score = brier_score_loss(y_test,   y_prob_test_cal1)
        print(f'Brier Score after calibration: {brier_score}')
        
        # Reshape calibrated probabilities
        y_prob_col = y_prob_test_cal1.reshape(-1, 1)
        y_prob_2col = np.hstack([1 - y_prob_col, y_prob_col])

        # Pass to roc_auc_binary
        print('After calibration')
        self.roc_auc_binary(y_test, y_prob_2col, label_binarizer.classes_)
        print('Before calibration')
        prob_true, prob_pred = calibration_curve(y_test,   y_prob_test[:, 1], n_bins=20,strategy='uniform')
        # Plot the calibration curve
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', color='green', markersize=4, label='Logistic Regression')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label = 'Ideally Calibrated')  # Diagonal line (perfect calibration)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Mean Predicted Probability', fontsize=18)
        plt.ylabel('Observed Frequency of Events', fontsize=18)
        plt.title('Calibration Curve', fontsize=18)
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
        print('After calibration')
        prob_true, prob_pred = calibration_curve(y_test,   y_prob_test_cal1, n_bins=5 ,strategy='uniform')
        
        # --- Hosmer–Lemeshow test function ---
        def hosmer_lemeshow_test(y_true, y_prob, g=10):
            """
            Perform Hosmer–Lemeshow test to assess calibration of predicted probabilities.

            Parameters:
            - y_true: array-like, true binary labels
            - y_prob: array-like, predicted probabilities
            - g: int, number of bins to group predictions (default=10)

            Returns:
            - hl_stat: Hosmer–Lemeshow test statistic
            - p_value: p-value for goodness-of-fit test
            """
            data = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
            data = data.sort_values('y_prob')
            data['bin'] = pd.qcut(data['y_prob'], g, duplicates='drop')

            obs = data.groupby('bin')['y_true'].sum()
            exp = data.groupby('bin')['y_prob'].sum()
            n = data.groupby('bin')['y_true'].count()

            exp_succ = exp
            exp_fail = n - exp
            obs_succ = obs
            obs_fail = n - obs

            hl_stat = np.sum(((obs_succ - exp_succ) ** 2) / (exp_succ + 1e-10) +
                             ((obs_fail - exp_fail) ** 2) / (exp_fail + 1e-10))

            p_value = 1 - chi2.cdf(hl_stat, g - 2)
            return hl_stat, p_value

        # --- Run the test ---
        hl_stat, p_val = hosmer_lemeshow_test(y_test, y_prob_test_cal1, g=10)
        print(f"Hosmer–Lemeshow Statistic: {hl_stat:.3f}")
        print(f"P-value: {p_val:.3f}")

        if p_val > 0.05:
            print("Good calibration (no significant difference between observed and predicted).")
        else:
            print("Poor calibration (predictions differ from observations).")
        
        
         # -------------------------------
         #  Calculate Expected Calibration Error (ECE)
        # -------------------------------
        def calculate_ece(y_true, y_prob, n_bins=10):
            """
            Calculate Expected Calibration Error (ECE)
            y_true: true labels (0 or 1)
            y_prob: predicted probabilities
            n_bins: number of bins to divide probability [0,1]
            """
            bins = np.linspace(0.0, 1.0, n_bins + 1)
            ece = 0.0
            n = len(y_true)

            for i in range(n_bins):
                # Bin range
                bin_lower = bins[i]
                bin_upper = bins[i + 1]
                # Find predictions in this bin
                mask = (y_prob > bin_lower) & (y_prob <= bin_upper)
                if np.any(mask):
                    prob_avg = y_prob[mask].mean()
                    true_avg = y_true[mask].mean()
                    ece += (np.sum(mask) / n) * np.abs(prob_avg - true_avg)
            return ece
        ece = calculate_ece(y_test, y_prob_test_cal1, n_bins=8)
        print("ECE:", ece)

        # Plot the calibration curve
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', color='green', markersize=4, label='Logistic Regression')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label = 'Ideally Calibrated')  # Diagonal line (perfect calibration)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Mean Predicted Probability', fontsize=18)
        plt.ylabel('Observed Frequency of Events', fontsize=18)
        plt.title('Calibration Curve', fontsize=18)
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
      
        # Create the explainer using the linear SHAP explainer
        explainer = shap.TreeExplainer(best_model)
        shap_values_test = explainer.shap_values(X_test)

        explainer = shap.TreeExplainer(best_model)
        shap_values = np.array(explainer.shap_values(X_test))
        print(shap_values.shape, shap_values_test, X_test.shape)
        #shap.summary_plot(shap_values[1],X_train, max_display=100)
        plt.rcParams.update({
            'font.size': 18,           # Base font size
            'axes.titlesize': 18,       # Font size for the axes title
            'axes.labelsize': 18,       # Font size for the axes labels
            'xtick.labelsize': 18,      # Font size for x-axis tick labels
            'ytick.labelsize': 18       # Font size for y-axis tick labels
        })
        output_folder = r"C:\Users\faith.neema\OneDrive - Aga Khan University\Documents\Longitudinal studies\Longitudinal data_prepared\SHAP_individual_figures"
        
        os.makedirs(output_folder, exist_ok=True)
        shap.summary_plot(shap_values,
                X_test, plot_type="bar", max_display=20, plot_size=(15, 10), show=False)#[:,:,1]to put infront of shap_values when rf is used


        ax = plt.gca()
        ax.set_xlabel(ax.get_xlabel(), fontsize=12)  # Modify size as needed fontweight='bold'
        for label in ax.get_yticklabels():
            #label.set_fontweight('bold')
            label.set_fontsize(12)  
        for label in ax.get_xticklabels():
           # label.set_fontweight('bold')
            label.set_fontsize(12)  

        # Display the updated plot
        #plt.suptitle("C. Feature importance plot by RF for off-track development", fontsize=34, y=1.02)

        # Save as separate PDF
        #plt.savefig(os.path.join(output_folder, "RF_feature_plot.pdf"), bbox_inches="tight",format='pdf',pad_inches=0)
        ##plt.close()
        #plt.show()
        plt.show()
        shap.summary_plot(shap_values,X_test, max_display=20, plot_size=(15, 10), show=False) #[:,:,1]to put infront of shap_values when rf is used

        #plt.rcParams['pdf.fonttype'] = 42
        ax = plt.gca()
        # Adjust font size of the x-axis label text
        ax.set_xlabel(ax.get_xlabel(), fontsize=12)  # Modify size as needed  fontweight='bold'
        ax.set_ylabel(ax.get_ylabel(), fontsize=12)  # Modify size as neededfontweight='bold'
        for label in ax.get_yticklabels():
            #label.set_fontweight('bold')
            label.set_fontsize(12)  
        for label in ax.get_xticklabels():
            #label.set_fontweight('bold')
            label.set_fontsize(12)  

                # Access the current color bar
        cbar = plt.gcf().axes[-1]  # The color bar is usually the last axis object in SHAP plots

        # Set the font size for color bar labels ('low' and 'high' feature values)
        cbar.tick_params(labelsize=12)  #
        # Set the font size for the color bar label (usually 'Feature value')
        cbar.set_ylabel(cbar.get_ylabel(), fontsize=12)  # Adjust font size and weight as needed, fontweight='bold'

        # Display the updated plot
       # plt.suptitle("A. Effect Plot by RF", fontsize=12, y=1.2)

        #plt.savefig(os.path.join(output_folder, "Effect Plot by RFnot editable.pdf"), bbox_inches="tight",pad_inches=0)
        #plt.close()
        plt.show()
        for i in range(43):
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(i, shap_values_test, X_test, interaction_index= None, show=False)#[:,:,1] #to put infront of shap_values_test when rf is used
            ax = plt.gca()
            ax.set_xlabel(ax.get_xlabel(), fontsize=8)  # Modify size as needed fontweight='bold'
            ax.set_ylabel(ax.get_ylabel(), fontsize=8)  # Modify size as needed fontweight='bold'
            for label in ax.get_yticklabels():
                #label.set_fontweight('bold')
                label.set_fontsize(8)  
            for label in ax.get_xticklabels():
                #label.set_fontweight('bold')
                label.set_fontsize(8)  

           # Display the updated plot
            plt.show()

        # explainer1 = shap.Explainer(best_model, X_train)
        # shap_values = explainer1(X_train)
        # shap.plots.waterfall(shap_values_test[0], check_additivity=False)
        #shap.dependence_plot( 'Child age (months)', shap_values_test[1], X_test, interaction_index=None, show=False)
        #shap.dependence_plot( 'Child height-for-age z-scores', shap_values_test[1], X_test, interaction_index=None, show=False)
        #shap_values_test[:, :, 1]
       # ax = plt.gca()
        #ax.set_xlabel(ax.get_xlabel(), fontsize=12)  # Modify size as needed fontweight='bold'
        #ax.set_ylabel(ax.get_ylabel(), fontsize=12)  # Modify size as needed fontweight='bold'
        #for label in ax.get_yticklabels():
            #label.set_fontweight('bold')
            #label.set_fontsize(12)  
        #for label in ax.get_xticklabels():
            # label.set_fontweight('bold')
           # label.set_fontsize(12)  

            # Display the updated plot
        #plt.suptitle("B. Child age Partial Dependence (PD) plot by RF" , fontsize=20, y=1.02)

        #plt.savefig(os.path.join(output_folder, "Child age_SHAP_Dependency.pdf"), bbox_inches="tight")
        #plt.close()
        #plt.suptitle("O. Child height for age Zscores PD plot by RF", fontsize=20, y=1.02)

        #plt.savefig(os.path.join(output_folder, "Child height-for-age_SHAP_Dependencynot editable.pdf"), bbox_inches="tight",pad_inches=0)
        #plt.close()
        ##plt.show()
        return best_model, y_pred_train, y_pred_test, y_prob_test, y_prob_train, shap_values_test, label_encoder 

