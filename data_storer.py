import numpy as np
import pandas as pd

from divide_data import divide_x, divide_y, divide_y_one_class

def creata_data_arrays():
    """
    A function that takes the original datasets (split into train/val/test) and creates the datasets for the classifier.
    The dataset for the classifier uses the manually designed features.
    The final datasets are stored as csv files in Data_Arrays folder.
    main.py reads from these csv files.
    """
    print("train data")
    with open('./Data/training_data.jsonl', 'r') as training_data_file:
        data = list(training_data_file)

    X_train = divide_x(data)
    y_train = divide_y(data)
    print("val data")
    with open('./Data/validation_data.jsonl', 'r') as validation_data_file:
        data = list(validation_data_file)

    X_validation = divide_x(data)
    y_validation = divide_y(data)

    print("test data")
    with open('./Data/test_data.jsonl', 'r') as test_data_file:
        data = list(test_data_file)

    X_test = divide_x(data)
    y_test = divide_y(data)
    y_test_phrase = divide_y_one_class(data, ['phrase'])
    y_test_passage = divide_y_one_class(data, ['passage'])
    y_test_multi = divide_y_one_class(data, ['multi'])

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_validation:", X_validation.shape)
    print("y_validation:", y_validation.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)
    print("y_test_phrase:", y_test_phrase.shape)
    print("y_test_passage:", y_test_passage.shape)
    print("y_test_multi:", y_test_multi.shape)

    # save data as csv
    df_X_train = pd.DataFrame(X_train)
    df_X_train.to_csv("./Data_Arrays/X_train.csv", index=False)

    df_y_train = pd.DataFrame(y_train)
    df_y_train.to_csv("./Data_Arrays/y_train.csv", index=False)

    df_X_validation = pd.DataFrame(X_validation)
    df_X_validation.to_csv("./Data_Arrays/X_validation.csv", index=False)

    df_y_validation= pd.DataFrame(y_validation)
    df_y_validation.to_csv("./Data_Arrays/y_validation.csv", index=False)

    df_X_test = pd.DataFrame(X_test)
    df_X_test.to_csv("./Data_Arrays/X_test.csv", index=False)

    df_y_test = pd.DataFrame(y_test)
    df_y_test.to_csv("./Data_Arrays/y_test.csv", index=False)

    df_y_test_phrase = pd.DataFrame(y_test_phrase)
    df_y_test_phrase.to_csv("./Data_Arrays/y_test_phrase.csv", index=False)

    df_y_test_passage = pd.DataFrame(y_test_passage)
    df_y_test_passage.to_csv("./Data_Arrays/y_test_passage.csv", index=False)

    df_y_test_multi = pd.DataFrame(y_test_multi)
    df_y_test_multi.to_csv("./Data_Arrays/y_test_multi.csv", index=False)

if __name__ == "__main__":
    creata_data_arrays()

