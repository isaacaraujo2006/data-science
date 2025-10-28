# 🔍 Refinamento LightGBM (Final Sênior)

## Melhores Hiperparâmetros
```json
{
    "boosting_type": "gbdt",
    "num_leaves": 113,
    "max_depth": 11,
    "learning_rate": 0.03724030002051435,
    "feature_fraction": 0.7436442469711639,
    "bagging_fraction": 0.7143900116442956,
    "bagging_freq": 2,
    "min_child_samples": 69,
    "lambda_l1": 0.8014088932838044,
    "lambda_l2": 4.747466293203226,
    "min_gain_to_split": 0.20923944564632804,
    "min_data_in_leaf": 78,
    "objective": "regression",
    "metric": [
        "rmse",
        "mae"
    ],
    "verbosity": -1,
    "random_state": 42
}
```

**Melhor iteração:** 744

## Métricas no Holdout
```json
{
    "RMSE": 0.983531489161883,
    "MAE": 0.5875487560627212,
    "R2": 0.7213825985501674,
    "MAPE": 38.84407064767768
}
```
## Importância das Features
| feature           |       importance |
|:------------------|-----------------:|
| discount          |      6.53596e+06 |
| temp_x_desc       |      1.57306e+06 |
| first_category_id | 958525           |
| avg_temperature   | 452925           |
| precpt            | 176949           |
| store_id          | 172480           |
| avg_humidity      | 147792           |
| weekday           | 133126           |
| lag_1             | 117068           |
| day               | 107807           |
| humid_x_act       |  78059.5         |
| rolling_mean_7    |  75108.1         |
| rolling_mean_14   |  46867.6         |
| rolling_mean_30   |  42163.8         |
| lag_30            |  30313.5         |
| lag_14            |  29417.1         |
| lag_7             |  28824           |
| activity_flag     |   8355.7         |
| month             |      0           |
| year              |      0           |

![feature_importance](feature_importance.png)


![dispersao_preditos](dispersao_preditos.png)
