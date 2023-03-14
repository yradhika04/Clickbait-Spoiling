
import argparse
from logistic_regression import LogReg
from divide_data import divide_x, divide_y, divide_y_one_class


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

    with open('test_data.jsonl', 'r') as test_data_file:
        data = list(test_data_file)

    X_test = divide_x(data)
    y_test = divide_y(data)
    y_test_phrase = divide_y_one_class(data, ['phrase'])
    y_test_passage = divide_y_one_class(data, ['passage'])
    y_test_multi = divide_y_one_class(data, ['multi'])

    args = parser.parse_args()

    if args.logreg_multi:
        #  how do we use the validation set?
        # Logistic regression model 1
        print("\nLogistic regression: multiclass")
        logistic_regression_model = LogReg(X_train, y_train, X_test, y_test)
        logistic_regression_model.train()
        y_pred = logistic_regression_model.predict()
        logistic_regression_model.accuracy_scores(y_pred)

    if args.logreg_ovr:
        # Logistic regression model 2
        print("\nLogistic regression: one-vs-rest [phrase] vs [passage, multi]")
        logistic_regression_model_phrase = LogReg(X_train, y_train, X_test, y_test_phrase)
        logistic_regression_model_phrase.train()
        y_pred = logistic_regression_model_phrase.predict()
        logistic_regression_model_phrase.accuracy_scores(y_pred)

        # Logistic regression model 3
        print("\nLogistic regression: one-vs-rest [passage] vs [phrase, multi]")
        logistic_regression_model_passage = LogReg(X_train, y_train, X_test, y_test_passage)
        logistic_regression_model_passage.train()
        y_pred = logistic_regression_model_passage.predict()
        logistic_regression_model_passage.accuracy_scores(y_pred)

        # Logistic regression model 4
        print("\nLogistic regression: one-vs-rest [multi] vs [phrase, passage]")
        logistic_regression_model_multi = LogReg(X_train, y_train, X_test, y_test_multi)
        logistic_regression_model_multi.train()
        y_pred = logistic_regression_model_multi.predict()
        logistic_regression_model_multi.accuracy_scores(y_pred)

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
