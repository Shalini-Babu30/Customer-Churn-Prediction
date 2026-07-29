import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("customer_churn_business_dataset.csv")

data = data[[
    "gender",
    "age",
    
    "weekly_active_days",
    "last_login_days_ago",
    "payment_method",
    "churn"
]]

label_encoder = LabelEncoder()

data["gender"] = label_encoder.fit_transform(data["gender"])
data["payment_method"] = label_encoder.fit_transform(data["payment_method"])

X = data.drop("churn", axis=1)
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

joblib.dump(model, "churn_model.pkl")

print("Model saved successfully!")