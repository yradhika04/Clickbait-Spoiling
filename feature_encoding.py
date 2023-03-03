
import spacy
import numpy as np
import re
# from spacy.cli import download
# download("en_core_web_sm)
# download("en_core_web_md")
# download("en_core_web_lg")
# download("en_core_web_trf")


def subset(original_list: list, subset_list: list):
    """
    The function checks if the pos tag sequence of a feature is a continuous subset of the pos tags of the entire
    sentence.
    :param original_list: list of the pos tags for all words in a sentence
    :param subset_list: list of pos tags in a feature
    :return: boolean value: True or False
    """
    is_subset = False
    for i in range(len(original_list) - len(subset_list) + 1):
        for j in range(len(subset_list)):
            if original_list[i + j] != subset_list[j]:
                break
        else:
            is_subset = True

    return is_subset


def check_feature(sentence: str):
    """
    The function checks the presence of features in the headlines and encodes them in a vector.
    :param sentence: original headline as a string
    :return: numpy array containing one hot encoded representation of the features
    """

    number_of_features = 56  # change acc to the final number of features
    array_for_features = np.zeros(number_of_features)

    # create a spacy object and then use it for pos tagging of the target sentence/headline
    spacy_object = spacy.load('en_core_web_sm')
    all_tokens_pos = spacy_object(sentence)

    pos_tags = [each_token.tag_ for each_token in all_tokens_pos]

    # phrase
    # feature 1 (numbering as on excel sheet)
    if subset(pos_tags, ['DT', 'JJS', 'NN']):
        array_for_features = np.insert(array_for_features, 0, 1)

    if subset(pos_tags, ['DT', 'JJS', 'NNS']):
        array_for_features = np.insert(array_for_features, 1, 1)

    # feature 2
    if subset(pos_tags, ['NNP', 'POS', 'JJS', 'JJ', 'NN']):
        array_for_features = np.insert(array_for_features, 2, 1)

    # feature 11
    if subset(pos_tags, ['PRP', 'MD', 'RB', 'VB']):
        array_for_features = np.insert(array_for_features, 3, 1)

    if subset(pos_tags, ['PRP', 'MD', 'RB', 'VB', 'WP']):
        array_for_features = np.insert(array_for_features, 4, 1)

    if subset(pos_tags, ['PRP', 'MD', 'RB', 'VB', 'DT']):
        array_for_features = np.insert(array_for_features, 5, 1)

    if subset(pos_tags, ['PRP', 'MD', 'RB', 'VB', 'WRB']):
        array_for_features = np.insert(array_for_features, 6, 1)

    # feature 12
    if subset(pos_tags, ['NNP', 'POS', 'JJS', 'JJ', 'NN']):
        array_for_features = np.insert(array_for_features, 7, 1)

    # feature 20
    if subset(pos_tags, ['DT', 'RBS', 'JJ', 'NN']):
        array_for_features = np.insert(array_for_features, 8, 1)

    # feature 21
    if subset(pos_tags, ['PRP$', 'JJ', 'NN']):
        array_for_features = np.insert(array_for_features, 9, 1)

    # feature 22
    if subset(pos_tags, ['CD', 'JJ', 'NN']):
        array_for_features = np.insert(array_for_features, 10, 1)

    # feature 23
    if subset(pos_tags, ['DT', 'NN']):
        array_for_features = np.insert(array_for_features, 11, 1)

    # feature 24 -> regex for @usernames
    pattern_exists = re.findall("@", sentence)
    if pattern_exists:
        array_for_features = np.insert(array_for_features, 12, 1)

    # passage
    # feature 5 -> 'here's why' occurs more often in passage than in multi
    if subset(pos_tags, ['DT', 'RBS', 'JJ', 'NN']):
        array_for_features = np.insert(array_for_features, 13, 1)

    # feature 6
    if subset(pos_tags, ['DT', 'VBZ', 'WRB']):
        array_for_features = np.insert(array_for_features, 14, 1)

    if subset(pos_tags, ['DT', 'VBZ', 'WRB', 'NN']):
        array_for_features = np.insert(array_for_features, 15, 1)

    if subset(pos_tags, ['DT', 'VBZ', 'WRB', 'PRP']):
        array_for_features = np.insert(array_for_features, 16, 1)

    # feature 7
    if subset(pos_tags, ['WRB']):
        array_for_features = np.insert(array_for_features, 17, 1)

    if subset(pos_tags, ['WRB', 'NNS']):
        array_for_features = np.insert(array_for_features, 18, 1)

    if subset(pos_tags, ['WBR', 'NNP']):
        array_for_features = np.insert(array_for_features, 19, 1)

    if subset(pos_tags, ['WBR', 'PRP$']):
        array_for_features = np.insert(array_for_features, 20, 1)

    # feature 8
    if subset(pos_tags, ['DT', 'VBZ', 'WP', 'VBZ']):
        array_for_features = np.insert(array_for_features, 21, 1)

    if subset(pos_tags, ['DT', 'VBZ', 'WP', 'VBN']):
        array_for_features = np.insert(array_for_features, 22, 1)

    if subset(pos_tags, ['DT', 'VBZ', 'WP', 'IN', 'VBZ']):
        array_for_features = np.insert(array_for_features, 23, 1)

    # feature 9 -> regex for 'scientist' and 'scientists'
    pattern_exists = re.findall('scientist|scientists', sentence)
    if pattern_exists:
        array_for_features = np.insert(array_for_features, 24, 1)

    # feature 10
    if subset(pos_tags, ['NN', 'PRP', 'VBD']):
        array_for_features = np.insert(array_for_features, 25, 1)

    if subset(pos_tags, ['NN', 'NN', 'VBZ']):
        array_for_features = np.insert(array_for_features, 26, 1)

    if subset(pos_tags, ['NNP', 'DT', 'NNP']):
        array_for_features = np.insert(array_for_features, 27, 1)

    if subset(pos_tags, ['NN', 'PRP', 'RB']):
        array_for_features = np.insert(array_for_features, 28, 1)

    if subset(pos_tags, ['NN', 'TO', 'RB']):
        array_for_features = np.insert(array_for_features, 29, 1)

    if subset(pos_tags, ['NN', 'TO', 'VB']):
        array_for_features = np.insert(array_for_features, 30, 1)

    if subset(pos_tags, ['NN', 'VBZ', 'JJ']):
        array_for_features = np.insert(array_for_features, 31, 1)

    if subset(pos_tags, ['NN', 'WRB']):
        array_for_features = np.insert(array_for_features, 32, 1)

    if subset(pos_tags, ['NN', 'IN', 'WRB']):
        array_for_features = np.insert(array_for_features, 33, 1)

    if subset(pos_tags, ['NN', 'IN', 'DT']):
        array_for_features = np.insert(array_for_features, 34, 1)

    if subset(pos_tags, ['NN', 'IN', 'PRP']):
        array_for_features = np.insert(array_for_features, 35, 1)

    # feature 13
    if subset(pos_tags, ['WP', 'VBZ']):
        array_for_features = np.insert(array_for_features, 36, 1)

    if subset(pos_tags, ['WP', 'VBD']):
        array_for_features = np.insert(array_for_features, 37, 1)

    if subset(pos_tags, ['WP', 'VBZ', 'IN']):
        array_for_features = np.insert(array_for_features, 38, 1)

    if subset(pos_tags, ['WP', 'VBZ', 'RB']):
        array_for_features = np.insert(array_for_features, 39, 1)

    if subset(pos_tags, ['WP', 'VBZ', 'WRB']):
        array_for_features = np.insert(array_for_features, 40, 1)

    # multi
    # feature 4
    if subset(pos_tags, ['CD', 'NN']):
        array_for_features = np.insert(array_for_features, 41, 1)

    if subset(pos_tags, ['CD', 'NNS']):
        array_for_features = np.insert(array_for_features, 42, 1)

    if subset(pos_tags, ['CD', 'JJ', 'NN']):
        array_for_features = np.insert(array_for_features, 43, 1)

    if subset(pos_tags, ['CD', 'JJ', 'NNS']):
        array_for_features = np.insert(array_for_features, 44, 1)

    if subset(pos_tags, ['CD', 'JJ', 'NN', 'NNS']):
        array_for_features = np.insert(array_for_features, 45, 1)

    if subset(pos_tags, ['CD', 'NN', 'NN', 'NNS']):
        array_for_features = np.insert(array_for_features, 46, 1)

    if subset(pos_tags, ['CD', 'NNP', 'NNS']):
        array_for_features = np.insert(array_for_features, 47, 1)

    # feature 15 -> regex for 'revealed + :"
    pattern_exists = re.findall("revealed:", sentence)
    if pattern_exists:
        array_for_features = np.insert(array_for_features, 48, 1)

    # feature 16 -> regex for ":" at the end of the sentence
    pattern_exists = re.findall(":$", sentence)
    if pattern_exists:
        array_for_features = np.insert(array_for_features, 49, 1)

    # feature 25
    pattern_exists = re.findall("[0-9]+\.", sentence)
    if pattern_exists:
        array_for_features = np.insert(array_for_features, 50, 1)

    pattern_exists = re.findall("[0-9]+,", sentence)
    if pattern_exists:
        array_for_features = np.insert(array_for_features, 51, 1)

    pattern_exists = re.findall("[0-9]+\)", sentence)
    if pattern_exists:
        array_for_features = np.insert(array_for_features, 52, 1)

    pattern_exists = re.findall("#[0-9]+", sentence)
    if pattern_exists:
        array_for_features = np.insert(array_for_features, 53, 1)

    # feature 17
    # feature 18 -> "How to" occurs in passage 18 times, and 16 times in multi
    # feature 19 -> regex or pos tags?

    else:
        pass

    return array_for_features
