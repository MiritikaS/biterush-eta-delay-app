# Databricks notebook source
# Bite Rush - Databricks Final Code
# Purpose: EDA, feature engineering, model training, evaluation, predictions and model export.

# COMMAND ----------

import math
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

DELAY_THRESHOLD_MINUTES = 30
RANDOM_STATE = 42

# COMMAND ----------

# Upload files to Databricks DBFS first:
# /FileStore/bite_rush/train.csv
# /FileStore/bite_rush/test.csv
# /FileStore/bite_rush/Sample_Submission.csv

TRAIN_PATH = "/dbfs/FileStore/bite_rush/train.csv"
TEST_PATH = "/dbfs/FileStore/bite_rush/test.csv"
SAMPLE_PATH = "/dbfs/FileStore/bite_rush/Sample_Submission.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_submission = pd.read_csv(SAMPLE_PATH)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Sample submission shape:", sample_submission.shape)

# COMMAND ----------

def clean_text_series(s):
    return (
        s.astype(str)
        .str.strip()
        .replace({"nan": np.nan, "NaN": np.nan, "NULL": np.nan, "": np.nan})
    )


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def pickup_delay_minutes(order_time, picked_time):
    order_dt = pd.to_datetime(order_time, errors="coerce", format="%H:%M:%S")
    picked_dt = pd.to_datetime(picked_time, errors="coerce", format="%H:%M:%S")
    diff = (picked_dt - order_dt).dt.total_seconds() / 60
    return diff.where(diff >= 0, diff + 1440)


def prepare_features(df, is_train=True):
    df = df.copy()

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = clean_text_series(df[col])

    df["Weatherconditions"] = df["Weatherconditions"].str.replace("conditions ", "", regex=False)

    numeric_cols = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
        "Vehicle_condition",
        "multiple_deliveries",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if is_train:
        df["Time_taken_minutes"] = (
            df["Time_taken(min)"].astype(str).str.extract(r"(\d+)")[0].astype(float)
        )
        df["Delay_flag"] = (df["Time_taken_minutes"] > DELAY_THRESHOLD_MINUTES).astype(int)
        df["Delay_status"] = np.where(df["Delay_flag"] == 1, "Delayed", "On-Time")

    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce", dayfirst=True)
    df["Order_hour"] = pd.to_datetime(df["Time_Orderd"], errors="coerce", format="%H:%M:%S").dt.hour
    df["Order_day_of_week"] = df["Order_Date"].dt.dayofweek
    df["Weekend_flag"] = df["Order_day_of_week"].isin([5, 6]).astype(int)
    df["Pickup_delay_minutes"] = pickup_delay_minutes(df["Time_Orderd"], df["Time_Order_picked"])

    df["Delivery_distance_km"] = haversine_km(
        df["Restaurant_latitude"],
        df["Restaurant_longitude"],
        df["Delivery_location_latitude"],
        df["Delivery_location_longitude"],
    )

    df["Rating_Band"] = pd.cut(
        df["Delivery_person_Ratings"],
        bins=[-np.inf, 4.0, 4.7, np.inf],
        labels=["Low Rating", "Medium Rating", "High Rating"],
    ).astype(object)

    df["Distance_Band"] = pd.cut(
        df["Delivery_distance_km"],
        bins=[-np.inf, 3, 8, np.inf],
        labels=["Short Distance", "Medium Distance", "Long Distance"],
    ).astype(object)

    return df


train_clean = prepare_features(train_df, is_train=True)
test_clean = prepare_features(test_df, is_train=False)

display(train_clean.head())

# COMMAND ----------

# Descriptive analytics KPIs
total_orders = len(train_clean)
delayed_orders = int(train_clean["Delay_flag"].sum())

summary_kpis = pd.DataFrame(
    {
        "KPI": [
            "Total Orders",
            "Average Delivery Time",
            "Median Delivery Time",
            "Minimum Delivery Time",
            "Maximum Delivery Time",
            "P90 Delivery Time",
            "Delayed Orders",
            "Delay Percentage",
            "On-Time Percentage",
            "Average Pickup Delay",
            "Average Delivery Distance",
        ],
        "Value": [
            total_orders,
            round(train_clean["Time_taken_minutes"].mean(), 2),
            round(train_clean["Time_taken_minutes"].median(), 2),
            round(train_clean["Time_taken_minutes"].min(), 2),
            round(train_clean["Time_taken_minutes"].max(), 2),
            round(train_clean["Time_taken_minutes"].quantile(0.90), 2),
            delayed_orders,
            round(delayed_orders / total_orders * 100, 2),
            round((total_orders - delayed_orders) / total_orders * 100, 2),
            round(train_clean["Pickup_delay_minutes"].mean(), 2),
            round(train_clean["Delivery_distance_km"].mean(), 2),
        ],
    }
)

display(summary_kpis)

# COMMAND ----------

# Descriptive and diagnostic grouped analysis
group_columns = [
    "City",
    "Road_traffic_density",
    "Weatherconditions",
    "Type_of_vehicle",
    "Type_of_order",
    "Festival",
    "multiple_deliveries",
    "Rating_Band",
    "Distance_Band",
]

eda_outputs = {}

for col in group_columns:
    temp = (
        train_clean.groupby(col, dropna=False)
        .agg(
            Total_Orders=("ID", "count"),
            Average_Delivery_Time=("Time_taken_minutes", "mean"),
            Median_Delivery_Time=("Time_taken_minutes", "median"),
            P90_Delivery_Time=("Time_taken_minutes", lambda x: x.quantile(0.90)),
            Delay_Percentage=("Delay_flag", lambda x: x.mean() * 100),
        )
        .reset_index()
    )
    for metric in [
        "Average_Delivery_Time",
        "Median_Delivery_Time",
        "P90_Delivery_Time",
        "Delay_Percentage",
    ]:
        temp[metric] = temp[metric].round(2)
    eda_outputs[col] = temp.sort_values("Average_Delivery_Time", ascending=False)
    print(f"EDA by {col}")
    display(eda_outputs[col])

# COMMAND ----------

# Correlation analysis
correlation_cols = [
    "Time_taken_minutes",
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",
    "Delivery_distance_km",
    "Pickup_delay_minutes",
    "Order_hour",
]

correlation_matrix = train_clean[correlation_cols].corr(numeric_only=True).round(3)
display(correlation_matrix)

# COMMAND ----------

# Predictive analytics: model training
target = "Time_taken_minutes"

feature_cols = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",
    "Delivery_distance_km",
    "Pickup_delay_minutes",
    "Order_hour",
    "Order_day_of_week",
    "Weekend_flag",
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
    "Rating_Band",
    "Distance_Band",
]

model_data = train_clean[feature_cols + [target]].dropna(subset=[target]).copy()
X = model_data[feature_cols]
y = model_data[target]

numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
categorical_features = [c for c in feature_cols if c not in numeric_features]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.001),
    "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        max_depth=16,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

model_results = []
trained_pipelines = {}

for name, model in models.items():
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_valid)

    mae = mean_absolute_error(y_valid, pred)
    rmse = math.sqrt(mean_squared_error(y_valid, pred))
    r2 = r2_score(y_valid, pred)
    mape = np.mean(np.abs((y_valid - pred) / y_valid)) * 100

    model_results.append(
        {
            "Model": name,
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 3),
            "MAPE": round(mape, 3),
        }
    )
    trained_pipelines[name] = pipeline

model_results_df = pd.DataFrame(model_results).sort_values("MAE")
display(model_results_df)

best_model_name = model_results_df.iloc[0]["Model"]
best_pipeline = trained_pipelines[best_model_name]
print("Best model selected:", best_model_name)

# COMMAND ----------

# Feature importance
final_model = best_pipeline.named_steps["model"]

if hasattr(final_model, "feature_importances_"):
    onehot = (
        best_pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["onehot"]
    )
    feature_names = numeric_features + onehot.get_feature_names_out(categorical_features).tolist()
    importance_df = (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": final_model.feature_importances_,
            }
        )
        .sort_values("Importance", ascending=False)
        .head(25)
    )
else:
    importance_df = pd.DataFrame(
        {"Note": [f"{best_model_name} does not provide tree-based feature importance."]}
    )

display(importance_df)

# COMMAND ----------

# Generate predictions for test data
test_features = test_clean[feature_cols].copy()
test_predictions = best_pipeline.predict(test_features)
test_predictions = np.maximum(test_predictions, 1)

prediction_output = pd.DataFrame(
    {
        "ID": test_clean["ID"],
        "Predicted_Time_taken_min": np.round(test_predictions, 2),
    }
)

prediction_output["Delay_Flag"] = (
    prediction_output["Predicted_Time_taken_min"] > DELAY_THRESHOLD_MINUTES
).astype(int)

prediction_output["Delay_Status"] = np.where(
    prediction_output["Delay_Flag"] == 1,
    "Delayed",
    "On-Time",
)

display(prediction_output.head(20))

# COMMAND ----------

# Save outputs to DBFS
output_dir = "/dbfs/FileStore/bite_rush_outputs"
dbutils.fs.mkdirs("/FileStore/bite_rush_outputs")

prediction_output.to_csv(f"{output_dir}/bite_rush_test_predictions.csv", index=False)
model_results_df.to_csv(f"{output_dir}/bite_rush_model_comparison.csv", index=False)
summary_kpis.to_csv(f"{output_dir}/bite_rush_descriptive_summary.csv", index=False)
importance_df.to_csv(f"{output_dir}/bite_rush_feature_importance.csv", index=False)

joblib.dump(
    {
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "feature_cols": feature_cols,
        "delay_threshold": DELAY_THRESHOLD_MINUTES,
    },
    f"{output_dir}/bite_rush_best_model.pkl",
)

print("Outputs saved to /FileStore/bite_rush_outputs/")

# COMMAND ----------

display(dbutils.fs.ls("/FileStore/bite_rush_outputs/"))
