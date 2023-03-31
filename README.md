# Clickbait Spoiler Classification - Final Project ANLP

## General Description

In this repository we present our final project for the course
"*Advanced Natural Language Processing (ANLP)"* in which we participated
during the winter semester 2022-2023 at Potsdam University. The
following students contributed in the development of this project:
Radhika Yadav, Dorothea MacPhail and Valentina Tretti.

**Spoiler Type Classification:**

For this project we chose task 5: *Clickbait Spoiling*, specifically
subtask 1: *Spoiler Type Classification*from [*SemEval 2023*](https://semeval.github.io/SemEval2023/tasks.html). The goal of this
task is to classify a dataset of clickbait posts into
classes (passage, phrase and multipart) according to the text type
that would be needed to generate a spoiler (Hagen et al. 2022)<sup>1</sup>.

## Repository Structure

### 1. Code

-   **Data:**

    - Original data sets (training and validation) in folder `datasets_original`

    - Includes the final data set splits (training,validation, and testing) in json format: `training_data.json`, `validation_data.json`, `test_data.json`

    - Includes the functions used for the data pre-processing: `train_test_split.py` (splits the original training data set (3,200 posts) into our training set (2,400) and test set (800)) and `update_validation_data_format.py` (used to store and read the validation set in the same json format we have the training and test set)

-  **Data Arrays:** includes all the data arrays created from the
        final dataset splits (train, validation and test) in the function
        `creata_data_arrays()` in `data_storer.py`. These arrays are
        stored as csv files in this folder and are then used by the
        classifiers. Includes a file `array_shapes.txt` specifying the shapes
        of all the arrays.

- `baseline.py`: used to create a naïve baseline that always
predicts the "phrase" class and provides the accuracy and balanced
accuracy scores of the baseline. 

- `create_patterns.py`: used to check the frequency of a given pattern in a post
headline of all classes.

- `data_storer.py`: contains the function `creata_data_arrays()` which creates different datasets (arrays) for the classifiers using the manually created features and stores them into csv files. All these csv files are available in `Data_Arrays` folder and are read from `main.py`. In addition, this function prints the shapes of all the created arrays.

- `divide_data.py`: used to create X and Y arrays for training and testing and called in `data_storer.py`. The `divide_x()` function creates the `X_data` array that contains all independent variables for training and testing. Then, `divide_y()` creates the `y_data` array composed of the dependent variables ("phrase", "passage" and "multi"). Finally, `divide_y_one_class()` creates the arrays composed of the dependent variables that need to be predicted ("phrase", "passage" and "multi") specifically for the one-vs-rest settings.

- `feature_encoding.py`: checks if a given POS tag sequence is a subset of the POS tags of an entire sentence and checks if a given feature is in a headline of a clickbait post. Called in `divide_data.py`.

- `logistic_regression.py`: used to train the logistic regression models (for both multi-class and one-vs-rest settings). Includes the functions to predict the labels, determine the top feature selection and calculate balanced accuracy scores. Called in `main.py`.

- `main.py`: contains the main scipt to load the data, train and test the classifiers. Detailed description of flags for running the script in the *Reproduce Results* section of this document.

- `pos_tag_list.txt`: includes the list of all English POS tags according to the SpaCy Library.

- `pos_tagging_spacy.ipynb`: notebook used to extract he POS tags of SpaCy library used for feature engineering.

- `read_data.py`: used to read the data from the datasets to determine possible feature patterns.

- `requirements.txt`: includes the packages used.

- `svm_classifier.py`: includes class used to train two types of SVM models (linear and non-linear) for both multi-class and one-vs-rest settings. Includes the following functions: `get_description()` used to print the description of the type of SVM model being used and the parameters; `clf.fit()` to fit the model; `get_prediction()` to predict the labels; `get_acc()` calculate balanced accuracy scores. Called in `main.py`.

- `svm_gridsearch.py`: used to conduct the hyperparameter tuning of the SVM models and returns the best parameters. Called in `main.py`.

### 2. Results:

- **Logreg_Results:**

    - Includes `Logreg_Results_overview.txt` which contains: accuracy results and feature scores for all settings with all features and only the best 20.

    - `Logreg_Results_plots.zip`: includes all plots for confusion matrix and graphs for best features.

- **SVM Results:**

    - Includes a file `SVM_Results_overview.txt` which contains all results for both linear and non-linear SVM models (for both multi and ovr settings) considering all features and only best 20 and 30 features.

    - Includes `svm_result_plots.zip` which contains to folders: `feature_20` and `feature_30` which have all the confusion matrix plots for accuracy results and the graphs with the chi square feature importance for each setting and model.

## Reproducing Results
### Get Started
Install all necessary packages with `requirements.txt`:
```
$ pip install Code/requirements.txt
```
In addition, install the SpaCy English language model needed for POS-tagging:
```
$ python -m spacy download en_core_web_sm
```
(More information on installing SpaCy models [here](https://spacy.io/usage))

### Reproduce the Dataset
The final datasets with features can be inspected in folder `Code/Data/Data_Arrays`.

The dataset is reproducable with the following steps: 
1. xreate train-test split
```
$ python Code/Data/train_test_split.py
```
2. Re-format the validation data
```
$ python Code/Data/update_validation_data_format.py
```
3. Create final datasets for all settings using the features
```
$ python data_storer.py
```

### Reproduce the Model Results
All models are trained and tested in `main.py`. Use different flags for running the different models in different settings:

1. train the logistic regression model in the multi-class setting, validate it with all the features, selecting best 20 features, re-training the model and measuring accuracy with validation and testing updated sets:
```
$ python Code/main.py --logreg_multi --prep_data
```

2. train the logistic regression model in the one-vs-rest setting, validate it with all the features, selecting best 20 features, re-training the model and measuring accuracy with validation and testing updated sets.
```
$ python Code/main.py --logreg_ovr --prep_data
```

3. train the both linear and non-linear svm models in the multi-class setting, validate it with all the features, selecting best 20 features, re-training the model and measuring accuracy with validation and testing updated sets.
```
$ python Code/main.py --svm_multi --prep_data
```

4. train the both linear and non-linear svm models in the one-vs-rest setting, validate it with all the features, selecting best 20 features, re-training the model and measuring accuracy with validation and testing updated sets.
```
$ python Code/main.py --svm_ovr --prep_data
```

## References
<sup>1</sup> Matthias Hagen, Maik Fröbe, Artur Jurk, and Martin Potthast. 2022. *Clickbait spoiling via question answering and passage retrieval.* In Proceedings of the
60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7025–7036, Dublin, Ireland. Association for Computational Linguistics.