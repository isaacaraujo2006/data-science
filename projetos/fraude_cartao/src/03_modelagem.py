import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Configurações de caminho
DATA_PATH = "D:/github/data-science/projetos/fraude_cartao/data/processed/processed.csv"
MODEL_DIR = "D:/github/data-science/projetos/fraude_cartao/models/"
os.makedirs(MODEL_DIR, exist_ok=True)

# Engenharia de features
def feature_engineering(df):
    df = df.copy()
    df['hour_of_day'] = ((df['time'] // 3600) % 24).astype(int)
    Q1 = df['amount'].quantile(0.25)
    Q3 = df['amount'].quantile(0.75)
    IQR = Q3 - Q1
    df['outlier_amount'] = ((df['amount'] < (Q1 - 1.5*IQR)) | (df['amount'] > (Q3 + 1.5*IQR))).astype(int)
    df['log_amount'] = np.log1p(df['amount'])
    return df

# Treinar, calibrar, avaliar e salvar modelo
def train_evaluate_save(model, model_name, X_train, y_train, X_test, y_test):
    print(f"\n🚀 Treinando e calibrando {model_name}...")
    calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    calibrated.fit(X_train, y_train)

    y_probs = calibrated.predict_proba(X_test)[:, 1]
    y_pred_default = (y_probs >= 0.5).astype(int)

    print(f"\n🔍 Avaliação padrão ({model_name}, threshold 0.5):")
    print(classification_report(y_test, y_pred_default, digits=4))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_probs):.4f}")

    # Threshold ótimo
    thresholds = np.linspace(0.01, 0.99, 99)
    f1_scores = [f1_score(y_test, (y_probs >= t).astype(int)) for t in thresholds]
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"\n🎯 Threshold ótimo: {best_threshold:.2f} com F1 = {best_f1:.4f}")

    y_pred_best = (y_probs >= best_threshold).astype(int)
    print(f"\n📊 Avaliação com threshold ótimo:")
    print(classification_report(y_test, y_pred_best, digits=4))

    # Salvar modelo e threshold
    joblib.dump(calibrated, os.path.join(MODEL_DIR, f"{model_name.lower()}_model.joblib"))
    with open(os.path.join(MODEL_DIR, f"{model_name.lower()}_threshold.txt"), 'w') as f:
        f.write(str(best_threshold))

    print(f"\n✅ {model_name} salvo em {MODEL_DIR}")
    return best_f1, best_threshold, model_name

def main():
    df = pd.read_csv(DATA_PATH)
    df = feature_engineering(df)

    X = df.drop(columns=['class'])
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    num_cols = ['time', 'amount', 'hour_of_day', 'log_amount']
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    smote = SMOTE(random_state=42, n_jobs=-1)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"\n🔄 SMOTE aplicado: {sum(y_train)} ➜ {sum(y_train_res)} fraudes")

    models = {
        "LogisticRegression": LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        "HistGB": HistGradientBoostingClassifier(max_iter=200, random_state=42)
    }

    results = []
    for name, model in models.items():
        best_f1, best_threshold, _ = train_evaluate_save(
            model, name, X_train_res, y_train_res, X_test, y_test
        )
        results.append((name, best_f1, best_threshold))

    print("\n🏆 Modelo com melhor F1:")
    best_model = max(results, key=lambda x: x[1])
    print(f"Modelo: {best_model[0]} | F1-score: {best_model[1]:.4f} | Threshold: {best_model[2]:.2f}")

if __name__ == "__main__":
    main()