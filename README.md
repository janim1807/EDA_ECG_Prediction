# Group 2 Biosignal 


## Group Members: 
- Forename: Jan
- Surname: Imhof

## Reproducibility

**Operating System**: Windows

**Python Version**: 3.9

**Environment Setup**: 
````
conda create -n bda python=3.9
conda activate bda
pip install -r requirements.txt
pip install .
````
**Data Structure**

Before running the code, add the ecg, eda and all_apps_wide data to the raw data folder (Need to be created) in data/raw, so it looks like the following:
```
raw/
├── 1/
│ ├── all_apps_wide-2024-05-16.csv
│ ├── ecg_results.csv
│ └── eda_results.csv
├── 2/
│ ├── all_apps_wide-2024-03-13.csv
│ ├── ecg_results.csv
│ └── eda_results.csv
├── 3/
│ ├── all_apps_wide-2024-03-13.csv
│ ├── ecg_results.csv
│ └── eda_results.csv
├── 4/
│ ├── all_apps_wide-2024-05-16.csv
│ ├── ecg_results.csv
│ └── eda_results.csv
...

```

**Main Entry Point**
````
cd src
python additional_features.py
python preprocessing.py
python train.py
python predict.py
````

**Unittest & docstring coverage**:
````
pytest --cov-report term --cov=src tests/
docstr-coverage src -i -f
````  
