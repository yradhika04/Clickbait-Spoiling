
import numpy as np
import json
from feature_encoding import check_feature


def divide_x(data: list):
    """
    Creates X_data- the numpy array of all independent variable(s) for training and testing separately
    :param data: list of dictionaries, where each dictionary is one post containing the headlines, text etc.
    :return: one numpy array
    """
    X_data = []
    for each_entry in data:
        all_posts = json.loads(each_entry)
        for each_post in all_posts:
            sentence = each_post["postText"][0]
            X_data.append(check_feature(sentence))

    X_data = np.array(X_data, dtype=object)
    X_data = X_data.astype(int)

    return X_data


def divide_y(data: list):
    """
    Creates y_data- the numpy array of dependent variable(s) that need to be predicted by our model
    :param data: list of dictionaries, where each dictionary is one post containing the headlines, text etc.
    :return: one numpy array
    """

    y_data = []  # array for all the labels
    for each_entry in data:
        all_posts = json.loads(each_entry)
        for each_post in all_posts:
            if each_post['tags'] == ['phrase']:
                y_data.append(0)
            elif each_post['tags'] == ['passage']:
                y_data.append(1)
            elif each_post['tags'] == ['multi']:
                y_data.append(2)

    y_data = np.array(y_data, dtype=object)
    y_data = y_data.astype(int)

    return y_data


def divide_y_one_class(data: list, class_label: list):
    """
    Creates y_data_one_class- the numpy array of dependent variable(s) that need to be predicted by our model, but
    specifically for one_vs_rest models as here only one class will be the desired label at a time
    :param data: list of dictionaries, where each dictionary is one post containing the headlines, text etc.
    :param class_label: the label which is the destied label for ovr
    :return: one numpy array
    """

    y_data = []  # array for all the labels
    for each_entry in data:
        all_posts = json.loads(each_entry)
        for each_post in all_posts:
            if each_post['tags'] == class_label:
                y_data.append(1)
            else:
                y_data.append(0)

    y_data = np.array(y_data, dtype=object)
    y_data = y_data.astype(int)

    return y_data

