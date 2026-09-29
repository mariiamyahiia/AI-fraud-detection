from pathlib import Path
import joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

df = pd.read_csv("data/raw/creditcard.csv").drop_duplicates()
X, y = df.drop(columns=["Class"]), df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

pre = ColumnTransformer(
    [("amount_scaler", RobustScaler(), ["Amount"]),
     ("time_scaler", StandardScaler(), ["Time"])],
    remainder="passthrough")
cols = ["Amount", "Time"] + [c for c in X.columns if c not in ["Amount", "Time"]]
Xtr = pd.DataFrame(pre.fit_transform(X_train), columns=cols, index=X_train.index)
Xte = pd.DataFrame(pre.transform(X_test), columns=cols, index=X_test.index)

Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("artifacts").mkdir(exist_ok=True)
Xtr.to_parquet("data/processed/X_train.parquet")
Xte.to_parquet("data/processed/X_test.parquet")
y_train.to_frame().to_parquet("data/processed/y_train.parquet")
y_test.to_frame().to_parquet("data/processed/y_test.parquet")
joblib.dump(pre, "artifacts/preprocessor.joblib")
print("Done:", Xtr.shape, Xte.shape)