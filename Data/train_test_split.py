# To split the provided training data (3200 posts) into our training data (2400 posts) and test data (800 posts)
# Don't need to run this again!

import json
import pandas as pd
from pandas import json_normalize
from sklearn.model_selection import train_test_split

with open('train.jsonl', 'r') as training_data_file:
    data = list(training_data_file)

all_data = []  # will be a list of dictionaries, each dictionary is a post
for each_entry in data:
    result = json.loads(each_entry)
    all_data.append(result)

# convert json to pandas dataframe
data_frame = json_normalize(all_data)

# split the data into two dataframes, test set is 25% of the training set
train, test = train_test_split(data_frame, test_size=0.25, stratify=data_frame['tags'])

# to check that the ratio of the distribution of classes is the same
print("Training data distribution \n", train['tags'].value_counts())
print("Test data distribution \n", test['tags'].value_counts())

# Save the dataframes to jsonl files
training_data = train.to_dict(orient='records')
with open('training_data.jsonl', 'w') as training_data_file:
    json.dump(training_data, training_data_file)

test_data = test.to_dict(orient='records')
with open('test_data.jsonl', 'w') as test_data_file:
    json.dump(test_data, test_data_file)
