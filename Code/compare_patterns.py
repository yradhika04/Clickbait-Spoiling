import json


def compare_pattern(pattern):
    """
    Basic function that checks for the given pattern in the post headlines of all three classes.
    It only checks word matching (not POS tags), it doesn't take position in text into account.
    It assumes all is lower case.
    
    Prints count of pattern per class and percentage (using count of samples per class in train set.)

    pattern: string

    """
    pattern = pattern.lower()

    # training set samples
    sample_dict = {"phrase": 1025, "passage": 956, "multi": 419}

    # pattern count
    pattern_dict = {"phrase": 0, "passage": 0, "multi": 0}
    pattern_pc_dict = {"phrase": 0, "passage": 0, "multi": 0}

    # get headline patterns
    with open('./Data/training_data.jsonl', 'r') as training_data_file:
        data = list(training_data_file)

    for each_entry in data:
        all_posts = json.loads(each_entry)
        for each_post in all_posts:
            if pattern in each_post['postText'][0].lower():
                pattern_dict[each_post['tags'][0]] += 1

    # get percentage and print
    for c in pattern_dict.keys():
        pattern_pc_dict[c] = int((pattern_dict[c] / sample_dict[c]) * 100)

        print(c, "\n", "count", pattern_dict[c], "%", pattern_pc_dict[c])
