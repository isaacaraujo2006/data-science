import pandas as pd
import numpy as np
import yaml
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ===============================
# 1️⃣ Leitura de configuração
# ===============================
config_path = r"D:/github/data-science/projetos/fraude_financeira/config/config.yaml"

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

dataset_path = config["data"]["processed_parquet"]
metrics_path = os.path.join(config["metrics"]["directory"], "resultado_modelos.parquet")

print(f"📂 Lendo dataset processado de: {dataset_path}")
df = pd.read_parquet(dataset_path)

# ===============================
# 2️⃣ Separar features e target
# ===============================
target_col = "fraude"
X = df.drop(columns=[target_col])
y = df[target_col]

# Identificar colunas categóricas para tratamento
cat_cols = X.select_dtypes(include=["object", "category"]).columns
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# ===============================
# 3️⃣ Dividir treino/teste
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===============================
# 4️⃣ Modelos e parâmetros
# ===============================
model_params = {
    "LogisticRegression": {
        "model": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
        ]),
        "params": {
            "clf__C": np.logspace(-3, 2, 6),
            "clf__solver": ["liblinear", "lbfgs"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(class_weight="balanced", random_state=42),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10]
        }
    },
    "LightGBM": {
        "model": LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1),
        "params": {
            "num_leaves": [31, 50, 100],
            "max_depth": [-1, 10, 20],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 300, 500]
        }
    },
    "CatBoost": {
        "model": CatBoostClassifier(
            class_weights=[1, 1], verbose=0, random_state=42
        ),
        "params": {
            "depth": [6, 8, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "iterations": [200, 500]
        }
    }
}

# ===============================
# 5️⃣ Treinar e avaliar
# ===============================
resultados = []

for nome, mp in model_params.items():
    print(f"\n🚀 Treinando {nome}...")
    search = RandomizedSearchCV(
        mp["model"],
        mp["params"],
        n_iter=5,
        scoring="f1",
        n_jobs=-1,
        cv=3,
        random_state=42
    )
    search.fit(X_train, y_train)

    # Melhor modelo
    best_model = search.best_estimator_

    # Previsões probabilísticas
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Encontrar melhor threshold para F1
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    print(f"🔍 Melhor threshold: {best_threshold:.4f}")

    # Previsões finais
    y_pred = (y_proba >= best_threshold).astype(int)

    # Métricas
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    resultados.append({
        "modelo": nome,
        "melhores_params": search.best_params_,
        "threshold": best_threshold,
        "AUC": auc,
        "F1": f1,
        "Recall": recall,
        "Precision": precision,
        "Matriz_Confusao": cm.tolist()
    })

# ===============================
# 6️⃣ Salvar resultados
# ===============================
resultados_df = pd.DataFrame(resultados)
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
resultados_df.to_parquet(metrics_path, index=False)

print(f"\n✅ Resultados salvos em: {metrics_path}")
print(resultados_df)
