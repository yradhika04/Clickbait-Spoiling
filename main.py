
import argparse
from divide_data import divide_x, divide_y, divide_y_one_class
from logistic_regression import LogReg


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

    with open('training_data.jsonl', 'r') as training_data_file:
        data = list(training_data_file)

    X_train = divide_x(data)
    y_train = divide_y(data)

    with open('validation_data.jsonl', 'r') as validation_data_file:
        data = list(validation_data_file)

    X_validation = divide_x(data)
    y_validation = divide_y(data)

    with open('test_data.jsonl', 'r') as test_data_file:
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
