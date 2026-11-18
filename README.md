```text
  ____                           ____        _        
 / ___|___ _ __  ___ _   _ ___  |  _ \  __ _| |_ __ _ 
| |   / _ \ '_ \/ __| | | / __| | | | |/ _` | __/ _` |
| |__|  __/ | | \__ \ |_| \__ \ | |_| | (_| | || (_| |
 \____\___|_| |_|___/\__,_|___/ |____/ \__,_|\__\__,_|
|  \/  (_)_ __ (_)_ __   __ _                         
| |\/| | | '_ \| | '_ \ / _` |                        
| |  | | | | | | | | | | (_| |                        
|_|  |_|_|_| |_|_|_| |_|\__, |                        
                        |___/                         
```
# Census Data Mining Project
## CSCI 5535
#### By: Nolan Busch  |   Andrew Doubrava  |  Brooke DeVaney | Jackson Fletcher
 
## Requirments:
1. Python 3.*
2. numpy
3. scikit-learn
4. matplotlib
5. xgboost

## Setup:
```bash
pip install -r requirments.txt
```

## Running:
```bash
python main -gaf    # Runs the graphs, models, and all K-Fold evaluations
python main -af     # Runs all models and all K-Fold evaluations
python main -a      # Runs all models, no K-Fold evaluations
python main -xf     # Runs only XGBoost with K-Fold evaluations
python main -x      # Runs only XGBoost, no K-Fold evaluations
python main -g      # Runs only the graphs
```

## Flags
* *-x for XGBoost*
* *-k for KNN*
* *-d for DTC*
* *-e for ETC*
* *-f to enable fold for selected model(s)*
* *-a to enable all models*
* *-g to enable all graphs
