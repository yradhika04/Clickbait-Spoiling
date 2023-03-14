
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns


class LogReg:
    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Initialises a logistic regression model
        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # defining a logistic regression model with sklearn
        self.logreg = LogisticRegression(multi_class='auto', solver='lbfgs')

    def train(self):
        """
        Trains a logistic regression model
        :return: None
        """
        # training the model
        self.logreg.fit(self.X_train, self.y_train)

    def predict(self):
        """
        Tests a logistics regression model and calculates the label predictions
        :return: y_pred- array of the model's predictions
        """
        # predicting labels using the test set
        y_pred = self.logreg.predict(self.X_test)
        return y_pred

    def accuracy_scores(self, y_pred):
        """
        Calculates the accuracy scores of both training and test data and visualizes the confusion matrix as a heat map
        :param y_pred: array of the model's predictions
        :return: None
        """
        # Calculating accuracy for both training and test data
        accuracy_training = self.logreg.score(self.X_train, self.y_train)
        accuracy_testing = self.logreg.score(self.X_test, self.y_test)
        print("Accuracy for training set: ", accuracy_training)
        print("Accuracy for testing set: ", accuracy_testing)

        # Calculating confusion matrix
        confusion_matrix = metrics.confusion_matrix(self.y_test, y_pred)
        print("Confusion Matrix: \n", confusion_matrix)

        # Visualization of the confusion matrix
        plt.figure(figsize=(9, 9))
        sns.heatmap(confusion_matrix, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
        plt.ylabel('Actual Clickbait Type')
        plt.xlabel('Predicted Clickbait Type')
        all_sample_title = 'Accuracy Score: {0}'.format(accuracy_testing)
        plt.title(all_sample_title, size=15)
        plt.show()
