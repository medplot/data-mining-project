# Medplot Data Mining Project

## Download Behavioral Risk Factor Surveillance System dataset
To download the brfss dataset you can use the following function if you want. 
The function requires that the kaggle python package is already installed.
You can create your personal kaggle api key under Account -> API -> Create New API Token.
If you download the brfss dataset manually please copy the `2015.csv` file into the `brfss_dataset` folder.
```
from preprocessing.preprocessing import download_brfss_dataset

download_brfss_dataset("kaggle_username", "your_kaggle_api_key")
```

## Load preprocessed dataset
To load the preprocessed dataset import and use the following function.
The function requires the dataset to be present in the `brfss_dataset` folder.
```
from preprocessing.preprocessing import get_preprocessed_brfss_dataset

brfss_dataset, brfss_target = get_preprocessed_brfss_dataset()
```