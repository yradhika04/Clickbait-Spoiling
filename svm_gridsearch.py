import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV

from sklearn.datasets import make_classification

class DoGridSearch():
    def __init__(self, parameters, linear=True) -> None:
        
        if linear:
            self.svm = LinearSVC()
            print("Running Linear SVM Classifier.")
        else:
            print("Running SVM Classifier with RBF kernel.")
            self.svm = SVC(kernel="rbf")
        
        # GridSearch uses all the data and does a strafied 10-fold cross-validation
        self.clf = GridSearchCV(self.svm, parameters, cv=10)
    
    def get_best_params(self, X, y):

        self.clf.fit(X, y)

        # if more than one set of params gets best score, only one will be returned
        best_param_dict = self.clf.best_params_
        best_param_idx = self.clf.best_index_
        best_mean_testscore = self.clf.cv_results_["mean_test_score"][best_param_idx]

        results = self.clf.cv_results_
        results_df = pd.DataFrame(results)

        return best_param_dict, best_mean_testscore, best_param_idx, results_df
    

if __name__ == "__main__":

    lin_svm_params = {'loss': ['hinge', 'squared_hinge'], 'penalty': ['l1', 'l2'],
                   'dual': [True, False], 
                   'C': [2 ** -15, 2 ** -13, 2 ** -11, 2 ** -9, 2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0,
                                        2 ** 1, 2 ** 3, 2 ** 5, 2 ** 6]}
    nonlin_svm_params = {'C': [2 ** -15, 2 ** -13, 2 ** -11, 2 ** -9, 2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0,
                                  2 ** 1, 2 ** 3, 2 ** 5, 2 ** 6],
                            'gamma': [2 ** -15, 2 ** -13, 2 ** -11, 2 ** -9, 2 ** -7, 2 ** -5, 2 ** -3, 2 ** -1, 2 ** 0,
                                      2 ** 1, 2 ** 3, 2 ** 5, 2 ** 6]}

    ######################################
    # CHANGE DATA HERE #
    
    # just for illustration    
    X, y = make_classification(n_features=4, random_state=0, n_classes=3, n_informative=3, n_redundant=1)

    #######################################

    # linear svm
    gs = DoGridSearch(lin_svm_params)
    lin_param_dict, lin_best_score, lin_param_idx, lin_results_df = gs.get_best_params(X,y)

    print("Linear SVM:")
    print("Best parameters:", lin_param_dict)
    print("Best mean test score:", lin_best_score)
    print("Best models' index in dataframe:", lin_param_idx)
    print()

    lin_results_df.to_csv("LinSVM_param_results.csv")

    # non-linear svm
    gs_nonlin = DoGridSearch(nonlin_svm_params, linear=False)
    nonlin_param_dict, nonlin_best_score, nonlin_param_idx, nonlin_results_df = gs_nonlin.get_best_params(X,y)

    print("Non-Linear SVM:")
    print("Best parameters:", nonlin_param_dict)
    print("Best mean test score:", nonlin_best_score)
    print("Best models' index in dataframe:", nonlin_param_idx)

    nonlin_results_df.to_csv("NonLinSVM_param_results.csv")



