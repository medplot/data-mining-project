# Medplot Data Mining Project

![Medplot CI pipeline](https://github.com/medplot/data-mining-project/actions/workflows/ci.yml/badge.svg)

## What is Medplot

Diabetes is one of the most common chronic diseases in many countries, with the United States leading the way. Diabetes
is a severe chronic disease in which individuals lose the ability to effectively regulate blood glucose levels.
Complications such as heart disease, vision loss, lower limb amputations, and kidney disease are associated with
chronically high blood glucose levels in diabetics. Thus, the disease can lead to a decreased quality of life and life
expectancy.

According to the Centers for Disease Control and Prevention (CDC), 34.2 million Americans have diabetes and 88 million
have prediabetes in 2018. In addition, the CDC estimates that 1 in 5 diabetics and about 8 in 10 prediabetics are
unaware of their risk. Diabetes also places a tremendous burden on the economy: The cost of diagnosed diabetes is
approximately 327 billion, and the total cost of undiagnosed diabetes and prediabetes is nearly 400 billion per
year.

The aim of the Medplot data mining project is primarily to identify patterns of which people are particularly
predisposed to
developing diabetes, so that prevention measures can be designed even better and in a more targeted manner.

## The  Behavioral Risk Factor Surveillance System dataset

We use data from the Behavioral Risk Factor Surveillance System (BRFSS). The BRFSS was established in 1984 by the
CDC, the national public health agency of the United States. The BRFSS collects data in all 50 U.S. States as well as in
the District of Columbia and three U.S. territories.

Being the largest continuously conducted health survey system in the world, the BRFSS conducts more than 400,000
interviews per year to collect data about health-related risk behaviors, chronic health conditions and the use of
preventive services. Examples of the data collected include tobacco use, health care coverage, physical activity and
fruit and vegetable consumption. We will use the dataset from 2015 for our analysis, it is provided as a CSV file. The
data provided by the BRFSS includes 330 factors for 441,456 interviewed individuals. A small subset of the dataset can
be viewed in the directory `brfss_dataset/2015_small.csv`. The complete dataset can be found
on [kaggle](https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system).

## Overview of the different approaches

We implement different classification approaches to test which approach performs best to discover patients who have
diabetes. For each of the following approaches we created a jupyter notebook that contains the data analysis with the
respective model. Functionality that is used in multiple notebooks is implemented separately and can be imported into
the notebooks.

### K-nearest neighbors

### K-nearest centroids

### Naive Bayes

### Decision Tree

### Random forest

### Support Vector Machine

### Artificial neural network

The neural network uses an embedding layer for the categorical input values, followed by a dropout layer. For the
numerical input values, a batch normalization is applied first. After these first steps the input values are concatenated
and passed into multiple linear layers.

## General functions for all notebooks

To install all packages that are required for the general functions you can use the `requirements.txt` file. Open a
terminal, navigate to the project root, activate the project environment and execute the following command to install
all dependencies from the `requirements.txt`file.

```
pip install -r requirements.txt
```

### Download the Behavioral Risk Factor Surveillance System dataset

To download the brfss dataset you can use the following function if you want.
The function requires that the kaggle python package is already installed.
You can create your personal kaggle api key under Account -> API -> Create New API Token.
If you download the brfss dataset manually please copy the `2015.csv` file into the `brfss_dataset` folder.

```
from preprocessing.preprocessing import download_brfss_dataset

download_brfss_dataset("kaggle_username", "your_kaggle_api_key")
```

### Load preprocessed dataset

To load the preprocessed dataset import and use the following function.
The function requires the dataset to be present in the `brfss_dataset` folder.

```
from preprocessing.preprocessing import get_preprocessed_brfss_dataset

brfss_dataset, brfss_target = get_preprocessed_brfss_dataset()
```

The file `preprocessing.py` contains further preprocessing functions, for example to get a train-test split, to
oversample the dataset or to get the target one hot encoded.

For the artificial neural network specific preprocessing functions are implemented in
the `neural_network_perprocessing.py` file. These preprocessing is needed in particular for the embedding layer of the
neural network.

### Ready-to-use plot functions

Plot functions can be found in the directory `visualization`. Currently the following plot functions are available:

- `classification_plots.py`
    - plot_decision_boundary (interesting for k-nearest classifiers)
    - plot_confusion_matrix
- `general_plots`
    - plot_class_frequencies (useful to show that the dataset is imbalanced)
- `neural_network_plots`
    - plot_loss
    - plot_multiple_loss_curves

## Continuous Integration pipeline

The CI pipeline is based on GitHub actions and consists of two steps.

- Flake8: This step checks the code linting and ensures that all python files follow the Python Enhancement Proposals.
  To execute flake8 locally open a terminal, navigate to the project root, ensure that the project environment where
  flake8 is installed is activated and execute the command `flake8`.
- Unit tests: Running the unit tests locally is similar to the flake8 execution but instead of executing the
  command `flake8` run the command `pytest`. If you are working in PyCharm you can also right-click on the
  test folder and select `Run pytest in test`.