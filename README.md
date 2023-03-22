# Venture Funding With Deep Learning

## Overview

This project's goal is to develop a deep leanrning model to predict whether an applicant will be successful if they receive funding from Alphabe Soup.

The data used for modeling consisted of details from 34299 organizations that previously obtained funding from Alphabet Soup. Their were 12 features and 1 target variable in the dataset:

* EIN
* NAME 
* CLASSIFICATION
* USE_CASE
* ORGANIZATION 
* STATUS
* INCOME_AMT 
* SPECIAL_CONSIDERATIONS
* ASK_AMT
* IS_SUCCESSFUL 

---

## Data Preprocessing

EIN and NAME were dropped from the data as they are identifiers for each company and do not contain relevant information for modeling purposes.

The categorical data fields ('APPLICATION_TYPE', 'AFFILIATION', 'CLASSIFICATION', 'USE_CASE', 'ORGANIZATION', 'SPECIAL_CONSIDERATIONS', 'INCOME_AMT') were encoded using OneHotEncoder and the subsequent dataframe was merged with remaining fields to create a new dataframe with 116 features and one target variable.

The dataframe was split into X (features) and y (target variable) dataframes, which were both then split into train and test datasets using scikit-learn train_test_split. 

A StandardScaler was then fit using the X_train data and then used to transform both X_train and X_test.

---

## Model Definitions

Once preprocessing was completed, a total of four deep learning models were trained and their respective evaluation metrics were compared. The four models were defined as follows:

#### Original Model

layer details:

| Type  | Units | Activation |
|-------|-------|------------|
| Dense | 60    | relu       |
| Dense | 30    | relu       |
| Dense | 1     | sigmoid    |


loss function: binary crossentropy
optimizer: adam
metrics: accuracy
epochs: 100


#### Alternative Model 1

layer details:

| Type  | Units | Activation |
|-------|-------|------------|
| Dense | 60    | relu       |
| Dense | 30    | relu       |
| Dense | 15    | relu       |
| Dense | 1     | sigmoid    |

loss function: binary crossentropy
optimizer: adam
metrics: accuracy
epochs: 100


#### Alternative Model 2

layer details:

| Type  | Units | Activation |
|-------|-------|------------|
| Dense | 60    | tanh       |
| Dense | 30    | tanh       |
| Dense | 1     | sigmoid    |


loss function: binary crossentropy
optimizer: adam
metrics: accuracy
epochs: 100


### Alternative Model 3

layer details:

| Type  | Units | Activation |
|-------|-------|------------|
| Dense | 60    | relu       |
| Dense | 30    | relu       |
| Dense | 1     | sigmoid    |


loss function: binary crossentropy
optimizer: adam
metrics: accuracy
epochs: 150

---

## Results

| Model               | Loss   | Accuracy |
|---------------------|--------|----------|
| Original Model      | 0.5622 | 0.7306   |
| Alternative Model 1 | 0.5928 | 0.7289   |
| Alternative Model 2 | 0.7459 | 0.4545   |
| Alternative Model 3 | 0.7100 | 0.4983   |

---

## Summary

The original model and alternative model 1 had an accuracy of ~0.73. Considering the original model is the simpler of the two, that would be the model out of the four that would be best to use for predictive purposes.

Further optimization should, however, be conducted and evaluations performed using metrics other than accuracy, such as recall, precision and AUC-ROC. 

---

## Saved Models

Copies of the four models are available in the SavedModels folder:

| Model               | Saved File      |
|---------------------|-----------------|
| Original Model      | AlphabetSoup.h5 |
| Alternative Model 1 | Alt1.h5         |
| Alternative Model 2 | Alt2.h5         |
| Alternative Model 3 | Alt3.h5         |

---


## Technologies

All code is contained within the included jupyter lab notebook. 

Python version 3.9.15 was used to complete this challenge and the following additional libraries were utilized:
* pandas
* numpy
* Tensorflow
* jupyterlab

---

## Installation Guide

A working python environment is required. The included environmnet.yml file can be used to create a conda environment:

```
conda env create -f environment.yml

```


Alternatively, an existing environment can be used in which case Jupyter Lab, numpy, pandas, hvplot, sqlalchemy and voilà must be installed if not already available. They may be installed using pip:

```
pip install jupyterlab
pip install numpy
pip install pandas
pip install tensorflow
```

---

## Usage

The included .ipynb file should be opened in Jupyter Lab, which can be started in the configured python environment

```
jupyter lab

```

Details on using Jupyter Lab are beyond the scope of this project. Please consult the [Jupyter Lab documentation](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html) for usage details.


---

## Contributors

Code framework provided by FinTech Bootcamp.
Code completion by Thomas L. Champion.

---

## License

License information can be found in the included LICENSE file.



