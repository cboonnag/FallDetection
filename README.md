
## Directory layout
    .
    ├── SisFall_dataset/                  
    ├── utilities    
    │   ├── preparation.py
    │   ├── model.py
    │   ├── benchmark.py
    │   └── threshold_moving.py  
    ├── models
    │   ├── exp1/
    │   ├── exp2/
    │   └── exp3/   
    ├── results
    │   ├── exp1/
    │   ├── exp2/
    │   └── exp3/
    ├── logs
    │   ├── exp1/
    │   │   └── exp1_MODEL_rate={}_discarded={}_log.txt
    │   ├── exp2/
    │   └── exp3/  
    ├── exp1.py
    ├── exp2.py
    ├── exp3.py
    └── README.md

## Logs
--- 
&nbsp;
&nbsp;

## Model Export 
---
FILENAME: model_MODEL_EXP_DISCARDED_FREQ_PARAM1_PARAM2.sav
&nbsp;
&nbsp;
## Result table
---
&nbsp;
FILENAME: result_MODEL_EXP_DISCARDED_FREQ_PARAM1_PARAM2.csv
| fold | param_1 | param_2 | f1   | acc  | se   | sp   | time |
|------|---------|---------|------|------|------|------|------|
| 3    | 2       | 4       | 0.43 | 0.88 | 0.85 | 0.62 | 32.1 |
| 4    | 2       | 4       | 0.52 | 0.84 | 0.78 | 0.74 | 38.5 |
| 5    | 2       | 4       | 0.37 | 0.90 | 0.88 | 0.67 | 31.3 |

FILENAME: result_MODEL_EXP_DISCARDED_FREQ_OVERALL.csv

| param_1 | param_2 | f1   | se_f1 | acc  | se_acc | se   | se_se | sp   | se_sp | time |   se_time |
|---------|---------|------|-------|------|--------|------|-------|------|-------|------|  ---------|
| 2       | 4       | 0.43 |       | 0.88 |        | 0.85 |       | 0.62 |       | 32.1 |           |
| 2       | 6       | 0.52 |       | 0.84 |        | 0.78 |       | 0.74 |       | 38.5 |           |
| 2       | 8       | 0.37 |       | 0.90 |        | 0.88 |       | 0.67 |       | 31.3 |           |