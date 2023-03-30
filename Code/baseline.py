import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score


def create_baseline():

    # load data: validation and test labels.
    y_validation = np.ravel(pd.read_csv("./Data_Arrays/y_validation.csv"))
    y_test = np.ravel(pd.read_csv("./Data_Arrays/y_test.csv"))

    # always predict "phrase" and get the balanced accuracy

    # "phrase" as label 0 in multiclass setting

    
    # validation set 
    n_val = y_validation.shape[0] # get length of label array
    y_hat_val = np.zeros(shape=n_val) # create array with all phrase
    acc_val = accuracy_score(y_validation, y_hat_val)
    accb_val = balanced_accuracy_score(y_validation, y_hat_val)

    # test
    n_test = y_test.shape[0]
    y_hat_test = np.zeros(shape=n_test)
    acc_test = accuracy_score(y_test, y_hat_test)
    accb_test = balanced_accuracy_score(y_test, y_hat_test)

    print("Accuracy score of baseline:")
    print("Val:     {:.3f}".format(acc_val))
    print("Test:    {:.3f}".format(acc_test))

    print()

    print("Balanced accuracy score of baseline:")
    print("Val:     {:.3f}".format(accb_val))
    print("Test:    {:.3f}".format(accb_test))


if __name__ == '__main__':
    create_baseline()








