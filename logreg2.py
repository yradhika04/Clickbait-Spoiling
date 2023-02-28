# Importing the class logistic regression
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

# Importing the train_test_split
from sklearn.model_selection import train_test_split

# Importing the metrics
from sklearn import metrics
from sklearn.metrics import classification_report
import pandas as pd  # Importing pandas
import numpy as np  # Importing numpy
import seaborn as sns  # Importing seaborn


""" 
--------------> IMPORTANT NOTES: 
            1) According to documentation on sklearn.linear_model.LogisticRegression the training algorithm uses the 
                one-vs-rest(OvR) scheme if the "multi_class" option is set to 'ovr' and uses cross-entropy loss if the 
                'multi_class' option is set to 'multinomial'
                

            2) Resources that could be useful whe putting together the POS tagging function with Logistic Regression: 
                A) POS tagging and Logistic regression from scratch:
                 https://github.com/oppasource/POS-tagging-using-Logistic-Regression-from-scratch/blob/master/Logistic_Regression_for_POS.ipynb
                B) Logistic regression with sklearn: 
                    a) https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
                    b) https://www.datacamp.com/tutorial/understanding-logistic-regression-python
                C) Log Reg One-vs-Rest: 
                    a) https://www.kaggle.com/code/satishgunjal/multiclass-logistic-regression-using-sklearn/notebook
                D) Log Reg Multi Class: 
                    a) https://machinelearningmastery.com/multinomial-logistic-regression-with-python/
    

"""
class LogReg:
    def __init__(self, ):
        """
        Input: vectors returned by POS Tagging Function
            Using sklearn Library to create logistic regression model.
        """
        self.input = input

# ---------------------------------------------******** STEP 0 ********---------------------------------------------
        """Here we should assign the vectors to X_train, y_train and the testing data to X_test and y_test.
        """


                                    # Logistic Regression One-vs-Rest
#---------------------------------------------******** STEP 1 ********---------------------------------------------
        # Instantiating the model using one-vs-rest
            # logreg =instance of the model
        logreg = LogisticRegression(multi_class='ovr', solver='liblinear'))

# ---------------------------------------------******** STEP 2 ********---------------------------------------------
    """ Fitting the model with the data
        METHOD- logreg.fit(self, X, y sample_weight)     
            X_train = headlines, keywords and text
            y_train = classes (phrase, passage and multi)
    """
    #Training the model on the data, the model is storing the information learned from the data.
        # Model is learning the relationship between clickbait posts (X_train) and labels- spoiler type (y_train)
        logreg.fit(X_train, y_train)

# ---------------------------------------------******** STEP 3 ********---------------------------------------------
     """Predicts class labels from samples in X (input).
        X: data matriz from which we want to get the predictions = test data set (input)
            - Shape of X: n_samples, n_classes

        Returns: y_pred (vector containing the class labels for each sample)
            - Shape of y_pred: n_samples,
        """
        # Making predictions on entire test data set.
        y_pred = logreg.predict(X_test)


# ---------------------------------------------******** STEP 4 ********---------------------------------------------
    # HEY!!!!! I am not sure if we have to implement this probability method, I think we only have to use the one above.

                    # Please check!!!!!!!

        """ Sklearn method- predict_prob(X):
           - Probability estimates.
           - The returned estimates for all classes are ordered by the label of classes.
                *** For a multi_class problem, if multi_class is set to be “multinomial” the softmax
                function is used to find the predicted probability of each class. Else use a one-vs-rest
                approach, i.e calculate the probability of each class assuming it to be positive using
                the logistic function. and normalize these values across all the classes.
        Input: array - shape: (n_samples,n_features)
            n_samples: number of samples
            n_features: number of features
        Returns: array -shape: (n_samples,n_classes)
            The probability of the sample for each class in the model, where classes are ordered as they are in self.classes_.
        """
        predictions = logreg.predict_prob(X_test)


# ---------------------------------------------******** STEP 5 ********---------------------------------------------
    # Measuring Model Performance
        # Recommendation: Check accuracy for both train and test data. The model should be good on both.
        """
        Input: training data (x and y) and test data (x and y)
                - We will calculate accuracy of the model for both training and testing data. Ideally, we should obtain
                a good accuracy on both.
        Returns: the mean accuracy on the given test data and labels.
                - Mean accuracy of self.predict(X) wrt. y.
        """
        # Calculating accuracy for training data set
        accuracy_training = logreg.score(X_train, y_train)

        # Calculating accuracy for testing data set
        accuracy_testing = logreg.score(X_test, y_test)

        print("Accuracy for training set: ", accuracy_training)
        print("Accuracy for testing set: ", accuracy_testing)


# ---------------------------------------------******** STEP 6 ********---------------------------------------------
    # Confusion Matrix

    #Creating the confusion matrix.
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
        print("Confusion Matrix: ", confusion_matrix)

# ---------------------------------------------******** STEP 7 ********---------------------------------------------
"""Code for the visualization comes from: https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
   Another resource for visualization: https://www.datacamp.com/tutorial/understanding-logistic-regression-python
   They also share a way of visualizing the confusion matrix with matplotlib we can also check that way of doing it. 
   """

    #Visualization of the confusion matrix with Seaborn.
        plt.figure(figsize = 9,9))
        sns.heatmap(confusion_matrix, annot = True, fmt=".3f", linewidths=.5, square = True,
        cmap = 'Blues_r');
        plt.ylabel('Actual Clickbait Type');
        plt.xlabel('Predicted Clickbait Type');
        all_sample_title = 'Accuracy Score: {0}'.format(score)
        plt.title(all_sample_title, size = 15);

# ---------------------------------------------******** STEP 8 ********---------------------------------------------
    # Evaluating the model using classification_report for accuracy, precision and recall.
        """ HEY!! I think we only discussed about evaluating the model with accuracy but I saw someone implementing 
            classification_report for calculating precision and recall, hence, adding these code lines in case they 
            are useful if not we just remove them. 
            Resource consulted: https://www.datacamp.com/tutorial/understanding-logistic-regression-python
        """
        target_names = ["phrase", "passage", "multi"]
        print(classification_report(y_test, y_pred, target_names=target_names))



# ---------------------------------------------********-----------********---------------------------------------------



                                    # Logistic Regression Multi Class
# RESOURCE: https: // michael - fuchs - python.netlify.app / 2019 / 11 / 15 / multinomial - logistic - regression /
#---------------------------------------------******** STEP 1 ********---------------------------------------------
        # Instantiating the model using multi class
# The multinomial logistic regression model will be fit using cross-entropy loss and will predict the integer value for each integer encoded class label.

        # NOTE: here I am puttin as solver 'newton-cg' but I saw others use 'saga' check what is the difference
        logreg = LogisticRegression(multi_class='multinomial', solver='newton-cg))

# ---------------------------------------------******** STEP 2 ********---------------------------------------------
    """ Fitting the model with the data
        METHOD- logreg.fit(self, X, y sample_weight)     
            X_train = headlines, keywords and text
            y_train = classes (phrase, passage and multi)
    """
    #Training the model on the data, the model is storing the information learned from the data.
        # Model is learning the relationship between clickbait posts (X_train) and labels- spoiler type (y_train)
        logreg.fit(X_train, y_train)

# ---------------------------------------------******** STEP 3 ********---------------------------------------------
     """Predicts class labels from samples in X (input).
        X: data matriz from which we want to get the predictions = test data set (input)
            - Shape of X: n_samples, n_classes

        Returns: y_pred (vector containing the class labels for each sample)
            - Shape of y_pred: n_samples,
        """
        # Making predictions on entire test data set.
        y_pred = logreg.predict(X_test)


# ---------------------------------------------******** STEP 4 ********---------------------------------------------
    # HEY!!!!! I am not sure if we have to implement this probability method, I think we only have to use the one above.

                    # Please check!!!!!!!

        """ Sklearn method- predict_prob(X):
           - Probability estimates.
           - The returned estimates for all classes are ordered by the label of classes.
                *** For a multi_class problem, if multi_class is set to be “multinomial” the softmax
                function is used to find the predicted probability of each class. Else use a one-vs-rest
                approach, i.e calculate the probability of each class assuming it to be positive using
                the logistic function. and normalize these values across all the classes.
        Input: array - shape: (n_samples,n_features)
            n_samples: number of samples
            n_features: number of features
        Returns: array -shape: (n_samples,n_classes)
            The probability of the sample for each class in the model, where classes are ordered as they are in self.classes_.
        """
        predictions = logreg.predict_prob(X_test)


# ---------------------------------------------******** STEP 5 ********---------------------------------------------
    # Measuring Model Performance
        # Recommendation: Check accuracy for both train and test data. The model should be good on both.
        """
        Input: training data (x and y) and test data (x and y)
                - We will calculate accuracy of the model for both training and testing data. Ideally, we should obtain
                a good accuracy on both.
        Returns: the mean accuracy on the given test data and labels.
                - Mean accuracy of self.predict(X) wrt. y.
        """
        # Calculating accuracy for training data set
        accuracy_training = logreg.score(X_train, y_train)

        # Calculating accuracy for testing data set
        accuracy_testing = logreg.score(X_test, y_test)

        print("Accuracy for training set: ", accuracy_training)
        print("Accuracy for testing set: ", accuracy_testing)


# ---------------------------------------------******** STEP 6 ********---------------------------------------------
    # Confusion Matrix

    #Creating the confusion matrix.
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
        print("Confusion Matrix: ", confusion_matrix)

# ---------------------------------------------******** STEP 7 ********---------------------------------------------
"""Code for the visualization comes from: https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
   Another resource for visualization: https://www.datacamp.com/tutorial/understanding-logistic-regression-python
   They also share a way of visualizing the confusion matrix with matplotlib we can also check that way of doing it. 
   """

    #Visualization of the confusion matrix with Seaborn.
        plt.figure(figsize = 9,9))
        sns.heatmap(confusion_matrix, annot = True, fmt=".3f", linewidths=.5, square = True,
        cmap = 'Blues_r');
        plt.ylabel('Actual Clickbait Type');
        plt.xlabel('Predicted Clickbait Type');
        all_sample_title = 'Accuracy Score: {0}'.format(score)
        plt.title(all_sample_title, size = 15);

# ---------------------------------------------******** STEP 8 ********---------------------------------------------
    # Evaluating the model using classification_report for accuracy, precision and recall.
        """ HEY!! I think we only discussed about evaluating the model with accuracy but I saw someone implementing 
            classification_report for calculating precision and recall, hence, adding these code lines in case they 
            are useful if not we just remove them. 
            Resource consulted: https://www.datacamp.com/tutorial/understanding-logistic-regression-python
        """
        target_names = ["phrase", "passage", "multi"]
        print(classification_report(y_test, y_pred, target_names=target_names))