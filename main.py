
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pdb

from divide_data import divide_x, divide_y, divide_y_one_class
from logistic_regression import LogReg
from svm_classifier import SVMClassifier
from svm_gridsearch import DoGridSearch


def main():
    parser = argparse.ArgumentParser(
        description='ANLP Clickbait Spoiling Task 1'
    )

    parser.add_argument(
        '--logreg_multi', dest='logreg_multi',
        help='Train the multiclass logistic regression model and calculate accuracy',
        action='store_true'
    )

    parser.add_argument(
        '--logreg_ovr', dest='logreg_ovr',
        help='Train three one-vs-rest logistic regression models and calculate accuracies',
        action='store_true'
    )

    parser.add_argument(
        '--svm_multi', dest='svm_multi',
        help='Train the multiclass svm models and calculate accuracy',
        action='store_true'
    )

    parser.add_argument(
        '--svm_ovr', dest='svm_ovr',
        help='Train three one-vs-rest svm models and calculate accuracies',
        action='store_true'
    )

    parser.add_argument(
        '--example', dest='random',
        help='For checking, delete before submission',
        action='store_true'
    )
    data_prepared = True
    if data_prepared:
        X_train = pd.read_csv("./Data_Arrays/X_train.csv").to_numpy()
        y_train = np.ravel(pd.read_csv("./Data_Arrays/y_train.csv"))
        X_validation = pd.read_csv("./Data_Arrays/X_validation.csv").to_numpy()
        y_validation = np.ravel(pd.read_csv("./Data_Arrays/y_validation.csv"))
        X_test = pd.read_csv("./Data_Arrays/X_test.csv").to_numpy()
        y_test = np.ravel(pd.read_csv("./Data_Arrays/y_test.csv"))
        y_test_phrase = np.ravel(pd.read_csv("./Data_Arrays/y_test_phrase.csv"))
        y_test_passage = np.ravel(pd.read_csv("./Data_Arrays/y_test_passage.csv"))
        y_test_multi = np.ravel(pd.read_csv("./Data_Arrays/y_test_multi.csv"))

    else:
        with open('./Data/training_data.jsonl', 'r') as training_data_file:
            data = list(training_data_file)

        X_train = divide_x(data)
        y_train = divide_y(data)

        with open('./Data/validation_data.jsonl', 'r') as validation_data_file:
            data = list(validation_data_file)

        X_validation = divide_x(data)
        y_validation = divide_y(data)

        with open('./Data/test_data.jsonl', 'r') as test_data_file:
            data = list(test_data_file)

        X_test = divide_x(data)
        y_test = divide_y(data)
        y_test_phrase = divide_y_one_class(data, ['phrase'])
        y_test_passage = divide_y_one_class(data, ['passage'])
        y_test_multi = divide_y_one_class(data, ['multi'])

    args = parser.parse_args()

    if args.logreg_multi:
        # Logistic regression model 1
        print("\nLogistic regression: multiclass")

        # train classifier on training data
        logistic_regression_model = LogReg(X_train, y_train)
        logistic_regression_model.train()

        # predict accuracy with validation data
        y_pred = logistic_regression_model.predict(X_validation)
        print("\nAccuracy values for 1. Training data and 2. Validation data")
        logistic_regression_model.accuracy_scores(y_pred, X_train, y_train, X_validation, y_validation)

        # update based on the most important features
        updated_X_train, updated_X_validation, updated_X_test = logistic_regression_model.top_feature_selection(X_validation, X_test)

        # train a new model on updated training data
        logistic_regression_model = LogReg(updated_X_train, y_train)
        logistic_regression_model.train()

        # predict accuracy with validation data again
        y_pred = logistic_regression_model.predict(updated_X_validation)
        print("\nAccuracy values for 1. Training data and 2. Validation data, after selecting 20 most important features")
        logistic_regression_model.accuracy_scores(y_pred, updated_X_train, y_train, updated_X_validation, y_validation)

        # predict accuracy with test data
        y_pred = logistic_regression_model.predict(updated_X_test)
        print("\nAccuracy values for 1. Training data and 2. Testing data")
        logistic_regression_model.accuracy_scores(y_pred, updated_X_train, y_train, updated_X_test, y_test)


    if args.logreg_ovr:
        # common steps for all 3 classes
        print("\nLogistic regression: one-vs-rest (3 models)")

        # train classifier on training data
        logistic_regression_model = LogReg(X_train, y_train)
        logistic_regression_model.train()
        y_pred = logistic_regression_model.predict(X_test)

        # predict accuracy with test data
        print("\nAccuracy values for 1. Training data and 2. Testing data")

        # Logistic regression model 2
        print("\n[phrase] vs [passage, multi]")
        logistic_regression_model.accuracy_scores(y_pred, X_train, y_train, X_test, y_test_phrase)

        # Logistic regression model 3
        print("\n[passage] vs [phrase, multi]")
        logistic_regression_model.accuracy_scores(y_pred, X_train, y_train, X_test, y_test_passage)

        # Logistic regression model 4
        print("\n[multi] vs [phrase, passage]")
        logistic_regression_model.accuracy_scores(y_pred, X_train, y_train, X_test, y_test_multi)

    if args.svm_multi:
        # SVM model 1
        print("\nSVM: multiclass")

        # HYPERPARAMETERS
        # do hyperparameter tuning as 10-fold cross-validation on training data
        print("-----------------------------------------------------------------")
        print("Hyperparameter tuning on 10-fold cross-validation")
        print("Warning: Some combinations of hyperparameters will raise errors or not converge.")
        print()
            
        # parameters to check for linear and non-linear SVM 
        """
        lin_svm_params = {'loss': ['hinge', 'squared_hinge'], 'penalty': ['l1', 'l2'],
                   'dual': [True, False], 
                   'C': [2 ** -15, 2 ** -13, 2 ** -11, 2 ** -9, 2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0,
                                        2 ** 1, 2 ** 3, 2 ** 5, 2 ** 6]}
        nonlin_svm_params = {'C': [2 ** -15, 2 ** -13, 2 ** -11, 2 ** -9, 2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0,
                                  2 ** 1, 2 ** 3, 2 ** 5, 2 ** 6],
                            'gamma': [2 ** -15, 2 ** -13, 2 ** -11, 2 ** -9, 2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0,
                                      2 ** 1, 2 ** 3, 2 ** 5, 2 ** 6]}
        """
        lin_svm_params = {'loss': ['hinge', 'squared_hinge'], 'penalty': ['l1', 'l2'],
                   'dual': [True, False], 
                   'C': [2 ** -15, 2 ** 6]}
        nonlin_svm_params = {'C': [2 ** -15, 2 ** 6],
                            'gamma': [2 ** -15, 2 ** 6]}
        
        # linear SVM hyperparameter tuning
        gs = DoGridSearch(lin_svm_params)
        lin_param_dict, lin_best_score, lin_param_idx, lin_results_df = gs.get_best_params(X_train,y_train)

        print("Linear SVM:")
        print("Best parameters:", lin_param_dict)
        print("Best mean test score:", lin_best_score)
        #print("Best models' index in dataframe:", lin_param_idx)
        print()

        # store full results table as csv
        #lin_results_df.to_csv("LinSVM_param_results.csv")

        # NON-LINEAR SVM
        gs_nonlin = DoGridSearch(nonlin_svm_params, linear=False)
        nonlin_param_dict, nonlin_best_score, nonlin_param_idx, nonlin_results_df = gs_nonlin.get_best_params(X_train,y_train)

        print("Non-Linear SVM:")
        print("Best parameters:", nonlin_param_dict)
        print("Best mean test score:", nonlin_best_score)
        #print("Best models' index in dataframe:", nonlin_param_idx)

        # store full results table as csv
        #nonlin_results_df.to_csv("NonLinSVM_param_results.csv")

        # TRAIN I: with all features and best hyperparams
        print("-----------------------------------------------------------------")
        print("Training with best hyperparameters")
        # linear SVM
        svm_lin = SVMClassifier(C=lin_param_dict["C"], linear=True, penalty=lin_param_dict["penalty"],
                                 loss=lin_param_dict["loss"], dual=lin_param_dict["dual"])
        svm_lin.fit_clf(X_train, y_train)
        svm_lin.get_description()
        print(" on training set with all features.")
        print()

        # non-linear SVM
        svm_nonlin = SVMClassifier(C=nonlin_param_dict["C"], linear=False, gamma=nonlin_param_dict["gamma"])
        svm_nonlin.fit_clf(X_train,y_train)
        svm_nonlin.get_description()
        print(" on training set with all features.")
        print()

        # VALIDATION I: get accuracy on validation set with all features
        print("-----------------------------------------------------------------")
        print("Validation using all features")
        y_pred_lin = svm_lin.get_prediction(X_validation) # linear
        y_pred_nonlin = svm_nonlin.get_prediction(X_validation) # non-linear
        print("\nAccuracy values for Validation data:")
        print("Linear SVM balanced accuracy score: {:.3f}".format(svm_lin.get_acc(y_validation, y_pred_lin, balanced=True)))
        print("Non-linear SVM balanced accuracy score: {:.3f}".format(svm_nonlin.get_acc(y_validation, y_pred_nonlin, balanced=True)))

        # FEATURE SELECTION
        # find best 20 features on training set
        print("-----------------------------------------------------------------")
        print("Feature Selection")
        feat_selector = SelectKBest(chi2, k=20)
        X_train_new = feat_selector.fit_transform(X_train, y_train)
        X_validation_new = feat_selector.transform(X_validation)
        X_test_new = feat_selector.transform(X_test)

        assert X_train_new.shape[1] == 20, "feature selection size mismatch"
        assert X_validation_new.shape[1] == 20, "feature selection size mismatch"
        assert X_test_new.shape[1] == 20, "feature selection size mismatch"

        # save info on most important features
        features_chosen = feat_selector.get_support(indices=True)
        feature_chi_scores = feat_selector.scores_
        print("20 important features indices:", features_chosen)
        print(feature_chi_scores)
        # create bar chart of feature importance
        plt.bar(x=[i for i in range(X_train.shape[1])], height=feature_chi_scores)
        plt.ylabel('Importance score')
        plt.xlabel('Feature number')
        plt.title('Chi square feature importance', size=15)
        plt.show()

        # TRAIN II
        # train models with reduced features on training set
        print("-----------------------------------------------------------------")
        print("Training with reducted features.")
        # linear SVM
        svm_lin2 = SVMClassifier(C=lin_param_dict["C"], linear=True, penalty=lin_param_dict["penalty"],
                                 loss=lin_param_dict["loss"], dual=lin_param_dict["dual"])
        svm_lin2.fit_clf(X_train_new, y_train)
        svm_lin2.get_description()
        print(" on training set with reduced features.")
        print()        

        # non-linear SVM
        svm_nonlin2 = SVMClassifier(C=nonlin_param_dict["C"], linear=False, gamma=nonlin_param_dict["gamma"])
        svm_nonlin2.fit_clf(X_train_new,y_train)
        svm_nonlin2.get_description()
        print(" on training set with reduced features.")
        print()

        # VALIDATION II: get accuracy on validation set with reduced features
        y_pred2_lin = svm_lin2.get_prediction(X_validation_new) # linear
        y_pred2_nonlin = svm_nonlin2.get_prediction(X_validation_new) # non-linear
        print("\nAccuracy values for Validation data, after selecting 20 most important features")
        print("Linear SVM balanced accuracy score: {:.3f}".format(svm_lin2.get_acc(y_validation, y_pred2_lin, balanced=True)))
        print("Non-linear SVM balanced accuracy score: {:.3f}".format(svm_nonlin2.get_acc(y_validation, y_pred2_nonlin, balanced=True)))
        print()
        
        # TEST: test models with reduced features on test set
        print("-----------------------------------------------------------------")
        y_pred2_lin_test = svm_lin2.get_prediction(X_test_new) # linear
        y_pred2_nonlin_test = svm_nonlin2.get_prediction(X_test_new) # non-linear
        print("\nAccuracy values on Test data")
        print("Linear SVM balanced accuracy score: {:.3f}".format(svm_lin2.get_acc(y_test, y_pred2_lin_test, balanced=True)))
        print("Non-linear SVM balanced accuracy score: {:.3f}".format(svm_nonlin2.get_acc(y_test, y_pred2_nonlin_test, balanced=True)))

        # confusion matrix. in absolute counts and as percentage.
        class_labels = ["phrase", "passage", "multi"]
        # for linear SVM
        cm_lin_abs = confusion_matrix(y_test, y_pred2_lin_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_lin_abs, display_labels=class_labels)
        disp.plot()
        plt.show()

        cm_lin_pc = confusion_matrix(y_test, y_pred2_lin_test, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_lin_pc, display_labels=class_labels)
        disp.plot()
        plt.show()

        # for non-linear SVM
        cm_nonlin_abs = confusion_matrix(y_test, y_pred2_nonlin_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_nonlin_abs, display_labels=class_labels)
        disp.plot()
        plt.show()

        cm_nonlin_pc = confusion_matrix(y_test, y_pred2_nonlin_test, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_nonlin_pc, display_labels=class_labels)
        disp.plot()
        plt.show()
        

    if args.svm_ovr:
        # SVM model 2
        print("\nSVM: one-vs-rest [phrase] vs [passage, multi]")

        # SVM model 3
        print("\nSVM: one-vs-rest [passage] vs [phrase, multi]")

        # SVM model 4
        print("\nSVM: one-vs-rest [multi] vs [phrase, passage]")

    if args.random:
        print("hello")


if __name__ == '__main__':
    main()
