# To conveniently see the data and then find features for all classes

import json

with open('./Data/training_data.jsonl', 'r') as training_data_file:
    data = list(training_data_file)

for each_entry in data:
    all_posts = json.loads(each_entry)
    for each_post in all_posts:
        # change the key and/or value entries here to see different parts of the data, e.g., we can see all the
        # clickbait headlines whose spoiler tag is 'passage' by replacing 'phrase' with 'passage' below
        if each_post['tags'] == ['phrase']:
            print(each_post["postText"])
