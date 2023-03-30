from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score


class SVMClassifier:
    def __init__(self, C=1.0, linear=True, gamma=None, loss=None, penalty=None, dual=None) -> None:
        """check sklearn documentation on SVC and LinearSVC for a description of the parameters"""
        
        self.c = C
        self.linear = linear
        self.gamma = gamma
        self.loss = loss
        self.penalty = penalty
        self.dual = dual

        if linear: # create linear SVM
            assert self.loss != None, "loss must be specified for linear SVM"
            assert self.penalty != None, "penalty must be specified for linear SVM"
            assert self.dual != None, "dual must be specified for linear SVM"
            
            self.clf = LinearSVC(penalty=self.penalty, loss=self.loss, dual=self.dual, C=self.c)
        
        else: # create non-linear SVM with radial basis function kernel
            assert self.gamma != None, "gamma must be specified for non-linear SVM"

            self.clf = SVC(kernel="rbf", C=self.c, gamma=self.gamma)
    
    def get_description(self):
        """function to print a description of the type of classifier and parameters"""
        
        if self.linear:
            print("Running LinearSVC with C={}, {} loss, {} penalty, and dual={}.".format(self.c, self.loss, self.penalty, self.dual))
        else:
            print("Running SVC with rbf kernel, C={} and gamma={}.".format(self.c, self.gamma))

    def fit_clf(self, X, Y):

        self.clf.fit(X,Y)

        return None
    
    def get_prediction(self, X):

        y_pred = self.clf.predict(X)

        return y_pred

    def get_acc(self, y_true, y_pred, balanced=False):

        """Get the accuracy score for the prediction. 
        Switch balance=True to get a balanced accuracy score for class imbalance."""
        
        if balanced:
            acc = balanced_accuracy_score(y_true, y_pred)
        else:
            acc = accuracy_score(y_true, y_pred)
        
        return acc


