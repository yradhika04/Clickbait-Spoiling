# to read and store the validation dataset in the same json format as our training and test sets

import json
from pandas import json_normalize

with open('validation.jsonl', 'r') as validation_data_file:
    data = list(validation_data_file)

all_data = []  # will be a list of dictionaries, each dictionary is a post
for each_entry in data:
    result = json.loads(each_entry)
    all_data.append(result)

# convert json to pandas dataframe
data_frame = json_normalize(all_data)

# Save the dataframe to jsonl file
validation_data = data_frame.to_dict(orient='records')
with open('validation_data.jsonl', 'w') as validation_data_file:
    json.dump(validation_data, validation_data_file)


