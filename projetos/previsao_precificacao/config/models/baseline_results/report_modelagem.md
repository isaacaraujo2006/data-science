# 🤖 Relatório de Modelagem Baseline

## Resultados Comparativos

| modelo       |     rmse |      mae |       r2 |        mape | melhores_parametros                                                                                              |   tempo_min |
|--------------|----------|----------|----------|-------------|------------------------------------------------------------------------------------------------------------------|-------------|
| LightGBM     | 0.704464 | 0.419996 | 0.782492 | 3.39355e+08 | {'num_leaves': 127, 'n_estimators': 500, 'min_child_samples': 20, 'learning_rate': 0.1, 'feature_fraction': 0.9} |     1.62277 |
| RandomForest | 0.713482 | 0.412556 | 0.776887 | 3.06491e+08 | {'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_depth': 20}                            |    17.7064  |
| CatBoost     | 0.726357 | 0.435949 | 0.768762 | 3.60496e+08 | {'learning_rate': 0.2, 'l2_leaf_reg': 7, 'iterations': 300, 'depth': 10}                                         |     3.54644 |

## Melhor Modelo
- **LightGBM**
- RMSE: 0.7045
- MAE: 0.4200
- R²: 0.7825
- MAPE: 339354748.68%

## Hiperparâmetros
```json
{
    "num_leaves": 127,
    "n_estimators": 500,
    "min_child_samples": 20,
    "learning_rate": 0.1,
    "feature_fraction": 0.9
}
```

## 📊 Comparativo Gráfico
![comparativo_rmse](comparativo_rmse.png)
