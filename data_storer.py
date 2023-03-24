import numpy as np

from divide_data import divide_x, divide_y, divide_y_one_class

def creata_data_arrays():
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

    
    # save the data as numpy objects
    with open('./Data_Arrays/X_train.npy', 'wb') as f1:
        np.save(f1, X_train)

    with open('./Data_Arrays/y_train.npy', 'wb') as f2:
        np.save(f2, y_train)

    with open('./Data_Arrays/X_validation.npy', 'wb') as f3:
        np.save(f3, X_validation)

    with open('./Data_Arrays/y_validation.npy', 'wb') as f4:
        np.save(f4, y_validation)

    with open('./Data_Arrays/X_test.npy', 'wb') as f5:
        np.save(f5, X_test)

    with open('./Data_Arrays/y_test.npy', 'wb') as f6:
        np.save(f6, y_test)

    with open('./Data_Arrays/y_test_phrase.npy', 'wb') as f7:
        np.save(f7, y_test_phrase)

    with open('./Data_Arrays/y_test_passage.npy', 'wb') as f8:
        np.save(f8, y_test_passage)
    
    with open('./Data_Arrays/y_test_multi.npy', 'wb') as f9:
        np.save(f9, y_test_multi)

if __name__ == "__main__":
    creata_data_arrays()

