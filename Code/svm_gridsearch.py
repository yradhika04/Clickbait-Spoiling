import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV

class DoGridSearch():
    def __init__(self, parameters, linear=True) -> None:
        """
        Class to perform GridSearch for an SVM classifier. 

        paramters: dict. Dictionary of parameters with parameter name as keys and list of parameter values as values.
        linear: bool. Set to false if non-linear SVM is needed. Default: True. 
        """
        
        if linear:
            self.svm = LinearSVC()
            print("Running Linear SVM Classifier.")
        else:
            print("Running SVM Classifier with RBF kernel.")
            self.svm = SVC(kernel="rbf")
        
        # GridSearch uses all the data and does a strafied 10-fold cross-validation
        self.clf = GridSearchCV(self.svm, parameters, cv=10)
    
    def get_best_params(self, X, y):
        """
        Method to fit classifier with different parameters. Returns information on best parameters.

        X: Dataset.
        y: gold labels of X.
        """

        self.clf.fit(X, y)

        # if more than one set of params gets best score, only one will be returned
        best_param_dict = self.clf.best_params_
        best_param_idx = self.clf.best_index_
        best_mean_testscore = self.clf.cv_results_["mean_test_score"][best_param_idx]

        # store full results in dataframe
        results = self.clf.cv_results_
        results_df = pd.DataFrame(results)

        return best_param_dict, best_mean_testscore, best_param_idx, results_df
