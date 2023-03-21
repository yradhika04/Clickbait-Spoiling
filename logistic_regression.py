from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest


class LogReg:
    def __init__(self, X_train, y_train):
        """
        Initialises a logistic regression model
        :param X_train
        :param y_train
        """
        self.X_train = X_train
        self.y_train = y_train

        # defining a logistic regression model with sklearn
        self.logreg = LogisticRegression(multi_class='auto', solver='lbfgs')

    def train(self):
        """
        Trains a logistic regression model
        :return: None
        """
        self.logreg.fit(self.X_train, self.y_train)

    def predict(self, X_data):
        """
        Tests a logistic regression model and calculates the label predictions
        :param X_data
        :return: y_pred- array of the model's predictions
        """
        y_pred = self.logreg.predict(X_data)
        return y_pred

    def top_feature_selection(self, X_validation, X_test):
        """
        Uses chi square values to choose top 20 features, plots them and also updates the independent variables' arrays
        :param X_validation
        :param X_test
        :return: three numpy arrays- updated_X_train, updated_X_validation, updated_X_test
        """
        top_features = SelectKBest(score_func=chi2, k=30)
        fit = top_features.fit(self.X_train, self.y_train)
        chi_scores = fit.scores_
        chi_scores = chi_scores.tolist()

        # updating arrays to reflect only top 20 important features
        updated_X_train = top_features.fit_transform(self.X_train, self.y_train)
        updated_X_validation = top_features.transform(X_validation)
        updated_X_test = top_features.transform(X_test)

        for i, v in enumerate(chi_scores):
            print('Feature no.: %0d, Score: %.3f' % (i+1, v))
            # plotting above results as a bar graph
        plt.bar([x for x in range(len(chi_scores))], chi_scores)
        plt.ylabel('Importance score')
        plt.xlabel('Feature number')
        plt.title('Chi square feature importance', size=15)
        plt.show()

        return updated_X_train, updated_X_validation, updated_X_test

    def accuracy_scores(self, y_pred, X_train, y_train, X_data, y_data):
        """
        Calculates the accuracy scores of both training and test data and visualizes the confusion matrix as a heat map
        :param y_pred: array of the model's predictions
        :param X_train
        :param y_train
        :param X_data
        :param y_data
        :return: None
        """
        # Calculating accuracy
        accuracy_training = self.logreg.score(X_train, y_train)
        accuracy_testing = self.logreg.score(X_data, y_data)
        print("Accuracy 1: ", accuracy_training)
        print("Accuracy 2: ", accuracy_testing)

        # Calculating confusion matrix
        confusion_matrix = metrics.confusion_matrix(y_data, y_pred)
        print("Confusion Matrix: \n", confusion_matrix)

        # Visualization of the confusion matrix
        plt.figure(figsize=(9, 9))
        sns.heatmap(confusion_matrix, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
        plt.ylabel('Actual Clickbait Type')
        plt.xlabel('Predicted Clickbait Type')
        all_sample_title = 'Accuracy Score: {0}'.format(accuracy_testing)
        plt.title(all_sample_title, size=15)
        plt.show()
