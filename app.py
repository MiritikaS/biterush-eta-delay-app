from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


APP_DIR = Path(__file__).resolve().parent
ASSET_DIR = APP_DIR / "assets"
LOGO_PATH = ASSET_DIR / "biterush_logo.png"
MODEL_PATH = APP_DIR / "bite_rush_best_model.pkl"
TRAIN_PATH = APP_DIR / "train.csv"
DEFAULT_DELAY_THRESHOLD = 30


st.set_page_config(page_title="Bite Rush Analytics", layout="wide")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #fff8ef 0%, #fff1df 45%, #ffffff 100%);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ff6b1a 0%, #f04a23 65%, #b42318 100%);
        }
        [data-testid="stSidebar"] * { color: white; }
        .hero {
            background: linear-gradient(105deg, #ff6b1a 0%, #ef3f23 55%, #ffd23f 125%);
            padding: 24px 28px;
            border-radius: 18px;
            color: white;
            box-shadow: 0 18px 45px rgba(239, 63, 35, .22);
            margin-bottom: 22px;
        }
        .hero h1 { margin: 0; font-size: 38px; font-weight: 900; }
        .hero p { margin: 8px 0 0 0; font-size: 17px; }
        .metric-card {
            background: white;
            border: 1px solid #fed7aa;
            border-left: 8px solid #ff6b1a;
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 10px 28px rgba(31,41,55,.08);
        }
        .metric-label {
            font-size: 12px;
            color: #6b7280;
            font-weight: 800;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 28px;
            color: #111827;
            font-weight: 900;
            margin-top: 4px;
        }
        .section-title {
            color: #111827;
            font-size: 25px;
            font-weight: 900;
            margin: 8px 0 12px 0;
        }
        .info-card {
            background: white;
            border: 1px solid #fed7aa;
            border-radius: 16px;
            padding: 18px;
            min-height: 130px;
            box-shadow: 0 10px 28px rgba(31,41,55,.08);
        }
        .info-card h3 { color: #ef3f23; margin: 0 0 8px 0; font-size: 19px; }
        .info-card p { color: #4b5563; margin: 0; line-height: 1.45; font-size: 14px; }
        .callout {
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-radius: 14px;
            padding: 15px 17px;
            color: #7c2d12;
            font-weight: 700;
        }
        .stButton > button {
            background: linear-gradient(90deg, #ff6b1a, #ef3f23);
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 900;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    left, right = st.columns([5, 1])
    with left:
        st.markdown(
            f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
            unsafe_allow_html=True,
        )
    with right:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=True)


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title: str, body: str) -> None:
    st.markdown(
        f'<div class="info-card"><h3>{title}</h3><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def clean_text(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .replace({"nan": np.nan, "NaN": np.nan, "NULL": np.nan, "": np.nan})
    )


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


@st.cache_data
def load_data() -> pd.DataFrame | None:
    if not TRAIN_PATH.exists():
        return None

    df = pd.read_csv(TRAIN_PATH)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = clean_text(df[col])

    df["Weatherconditions"] = df["Weatherconditions"].str.replace("conditions ", "", regex=False)
    df["Time_taken_minutes"] = df["Time_taken(min)"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    df["Delay_Status"] = np.where(df["Time_taken_minutes"] > DEFAULT_DELAY_THRESHOLD, "Delayed", "On-Time")
    df["Delay_Flag"] = (df["Delay_Status"] == "Delayed").astype(int)

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

    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce", dayfirst=True)
    order_time = pd.to_datetime(df["Time_Orderd"], errors="coerce", format="%H:%M:%S")
    picked_time = pd.to_datetime(df["Time_Order_picked"], errors="coerce", format="%H:%M:%S")
    df["Order_hour"] = order_time.dt.hour
    df["Order_day_of_week"] = df["Order_Date"].dt.dayofweek
    df["Weekend_flag"] = df["Order_day_of_week"].isin([5, 6]).astype(int)
    pickup_delay = (picked_time - order_time).dt.total_seconds() / 60
    df["Pickup_delay_minutes"] = pickup_delay.where(pickup_delay >= 0, pickup_delay + 1440)

    df["Delivery_distance_km"] = haversine_km(
        df["Restaurant_latitude"],
        df["Restaurant_longitude"],
        df["Delivery_location_latitude"],
        df["Delivery_location_longitude"],
    )

    df["Rating_Band"] = pd.cut(
        df["Delivery_person_Ratings"],
        [-np.inf, 4.0, 4.7, np.inf],
        labels=["Low Rating", "Medium Rating", "High Rating"],
    ).astype(object)
    df["Distance_Band"] = pd.cut(
        df["Delivery_distance_km"],
        [-np.inf, 3, 8, np.inf],
        labels=["Short Distance", "Medium Distance", "Long Distance"],
    ).astype(object)
    return df


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### Dashboard Filters")
    city = st.sidebar.multiselect("City Type", sorted(df["City"].dropna().unique()), default=sorted(df["City"].dropna().unique()))
    traffic = st.sidebar.multiselect("Traffic Density", sorted(df["Road_traffic_density"].dropna().unique()), default=sorted(df["Road_traffic_density"].dropna().unique()))
    weather = st.sidebar.multiselect("Weather Condition", sorted(df["Weatherconditions"].dropna().unique()), default=sorted(df["Weatherconditions"].dropna().unique()))
    vehicle = st.sidebar.multiselect("Vehicle Type", sorted(df["Type_of_vehicle"].dropna().unique()), default=sorted(df["Type_of_vehicle"].dropna().unique()))
    order_type = st.sidebar.multiselect("Order Type", sorted(df["Type_of_order"].dropna().unique()), default=sorted(df["Type_of_order"].dropna().unique()))
    festival = st.sidebar.multiselect("Festival", sorted(df["Festival"].dropna().unique()), default=sorted(df["Festival"].dropna().unique()))

    min_date, max_date = df["Order_Date"].min(), df["Order_Date"].max()
    date_range = st.sidebar.date_input("Order Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        start_date, end_date = min_date, max_date

    return df[
        df["City"].isin(city)
        & df["Road_traffic_density"].isin(traffic)
        & df["Weatherconditions"].isin(weather)
        & df["Type_of_vehicle"].isin(vehicle)
        & df["Type_of_order"].isin(order_type)
        & df["Festival"].isin(festival)
        & df["Order_Date"].between(start_date, end_date)
    ].copy()


def repair_sklearn_tree_attributes(model_obj):
    """Repair old scikit-learn pickles that miss newer tree attributes."""
    if model_obj is None:
        return model_obj

    if not hasattr(model_obj, "monotonic_cst"):
        try:
            model_obj.monotonic_cst = None
        except Exception:
            pass

    if hasattr(model_obj, "estimators_"):
        for estimator in model_obj.estimators_:
            repair_sklearn_tree_attributes(estimator)

    if hasattr(model_obj, "steps"):
        for _, step in model_obj.steps:
            repair_sklearn_tree_attributes(step)

    return model_obj


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def train_lightweight_model_from_csv():
    df = load_data()
    if df is None:
        return None

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

    model_df = df[feature_cols + ["Time_taken_minutes"]].dropna(subset=["Time_taken_minutes"]).copy()
    X = model_df[feature_cols]
    y = model_df["Time_taken_minutes"]

    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_onehot_encoder()),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    pipeline.fit(X, y)
    return {
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "delay_threshold": DEFAULT_DELAY_THRESHOLD,
        "model_name": "Ridge Regression Live Deployment Model",
    }


@st.cache_resource
def load_model_bundle():
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            if not isinstance(bundle, dict) or "pipeline" not in bundle or "feature_cols" not in bundle:
                repaired = repair_sklearn_tree_attributes(bundle)
                return {"pipeline": repaired, "feature_cols": None, "delay_threshold": DEFAULT_DELAY_THRESHOLD, "model_name": "Uploaded Model"}
            bundle["pipeline"] = repair_sklearn_tree_attributes(bundle["pipeline"])
            bundle.setdefault("delay_threshold", DEFAULT_DELAY_THRESHOLD)
            bundle.setdefault("model_name", "Uploaded Model")
            return bundle
        except Exception:
            pass

    return train_lightweight_model_from_csv()


def page_eda(df: pd.DataFrame | None) -> None:
    hero("Bite Rush EDA Analysis", "Descriptive, diagnostic and predictive analytics view for delivery performance.")
    a, b, c = st.columns(3)
    with a:
        info_card("Descriptive Analytics", "Understand order volume, delivery time, delay percentage, city, traffic, weather and vehicle patterns.")
    with b:
        info_card("Diagnostic Analytics", "Identify delay drivers such as traffic, weather, festival period, distance, pickup delay and multiple deliveries.")
    with c:
        info_card("Predictive Analytics", "Predict estimated delivery time and classify each order as On-Time or Delayed.")

    if df is None:
        st.warning(f"Dataset not found at: {TRAIN_PATH}")
        return

    filtered = filter_data(df)
    st.markdown('<div class="section-title">EDA Data Preview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Filtered Orders", f"{len(filtered):,.0f}")
    with c2:
        metric_card("Avg Delivery Time", f"{filtered['Time_taken_minutes'].mean():.1f} min")
    with c3:
        metric_card("Delay %", f"{filtered['Delay_Flag'].mean() * 100:.1f}%")
    with c4:
        metric_card("Avg Distance", f"{filtered['Delivery_distance_km'].mean():.2f} km")
    st.markdown(
        '<div class="callout">Use the Live Dashboards page for full charts and decomposition analysis.</div>',
        unsafe_allow_html=True,
    )


def show_dashboard(df: pd.DataFrame, title: str, key_prefix: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if df.empty:
        st.error("No records match the selected filters.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Orders", f"{len(df):,.0f}")
    with c2:
        metric_card("Average Delivery Time", f"{df['Time_taken_minutes'].mean():.1f} min")
    with c3:
        metric_card("Delay Percentage", f"{df['Delay_Flag'].mean() * 100:.1f}%")
    with c4:
        metric_card("P90 Delivery Time", f"{df['Time_taken_minutes'].quantile(.9):.1f} min")

    left, right = st.columns([1.08, 1])
    with left:
        traffic_summary = (
            df.groupby("Road_traffic_density", dropna=False)
            .agg(Average_Delivery_Time=("Time_taken_minutes", "mean"))
            .reset_index()
        )
        fig = px.bar(
            traffic_summary,
            x="Road_traffic_density",
            y="Average_Delivery_Time",
            title="Average Delivery Time by Traffic Density",
            color="Average_Delivery_Time",
            color_continuous_scale=["#ffd23f", "#ff6b1a", "#ef3f23"],
        )
        fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_traffic")

    with right:
        donut_data = df["Delay_Status"].value_counts().reset_index()
        donut_data.columns = ["Delay_Status", "Orders"]
        fig = px.pie(
            donut_data,
            names="Delay_Status",
            values="Orders",
            hole=0.55,
            title="On-Time vs Delayed Orders",
            color="Delay_Status",
            color_discrete_map={"On-Time": "#0f766e", "Delayed": "#ef3f23"},
        )
        fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_donut")

    left, right = st.columns([1, 1])
    with left:
        trend = (
            df.groupby("Order_Date", dropna=False)
            .agg(Average_Delivery_Time=("Time_taken_minutes", "mean"), Orders=("ID", "count"))
            .reset_index()
            .sort_values("Order_Date")
        )
        fig = px.line(trend, x="Order_Date", y="Average_Delivery_Time", markers=True, title="Delivery Time Trend by Date")
        fig.update_traces(line_color="#ff6b1a")
        fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_trend")

    with right:
        scatter_sample = df.sample(min(len(df), 5000), random_state=42)
        fig = px.scatter(
            scatter_sample,
            x="Delivery_distance_km",
            y="Time_taken_minutes",
            color="Road_traffic_density",
            title="Distance vs Delivery Time",
            opacity=0.6,
        )
        fig.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_scatter")

    st.markdown('<div class="section-title">Decomposition Tree Style View</div>', unsafe_allow_html=True)
    tree_df = (
        df.groupby(["City", "Road_traffic_density", "Weatherconditions"], dropna=False)
        .agg(Delayed_Orders=("Delay_Flag", "sum"), Total_Orders=("ID", "count"))
        .reset_index()
    )
    fig = px.sunburst(
        tree_df,
        path=["City", "Road_traffic_density", "Weatherconditions"],
        values="Total_Orders",
        color="Delayed_Orders",
        color_continuous_scale=["#ffd23f", "#ff6b1a", "#ef3f23"],
        title="Delay Decomposition by City, Traffic and Weather",
    )
    fig.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_sunburst")


def show_delay_dashboard(df: pd.DataFrame) -> None:
    show_dashboard(df, "Delay Analysis Dashboard", "delay")
    weather = (
        df.groupby("Weatherconditions", dropna=False)
        .agg(Delay_Percentage=("Delay_Flag", lambda x: x.mean() * 100))
        .reset_index()
    )
    fig = px.bar(
        weather,
        y="Weatherconditions",
        x="Delay_Percentage",
        orientation="h",
        title="Delay % by Weather Condition",
        color="Delay_Percentage",
        color_continuous_scale=["#ffd23f", "#ff6b1a", "#ef3f23"],
    )
    fig.update_layout(height=390, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, key="delay_weather_bar")


def show_partner_dashboard(df: pd.DataFrame) -> None:
    if df.empty:
        st.error("No records match the selected filters.")
        return
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Average Partner Rating", f"{df['Delivery_person_Ratings'].mean():.2f}")
    with c2:
        metric_card("Average Pickup Delay", f"{df['Pickup_delay_minutes'].mean():.1f} min")
    with c3:
        metric_card("Average Distance", f"{df['Delivery_distance_km'].mean():.2f} km")
    with c4:
        metric_card("Average Delivery Time", f"{df['Time_taken_minutes'].mean():.1f} min")

    left, right = st.columns(2)
    with left:
        vehicle = (
            df.groupby("Type_of_vehicle", dropna=False)
            .agg(Average_Delivery_Time=("Time_taken_minutes", "mean"))
            .reset_index()
        )
        fig = px.bar(
            vehicle,
            x="Type_of_vehicle",
            y="Average_Delivery_Time",
            title="Average Delivery Time by Vehicle Type",
            color="Average_Delivery_Time",
            color_continuous_scale=["#ffd23f", "#ff6b1a", "#ef3f23"],
        )
        fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, key="partner_vehicle_bar")

    with right:
        rating = (
            df.groupby("Rating_Band", dropna=False)
            .agg(Average_Delivery_Time=("Time_taken_minutes", "mean"), Orders=("ID", "count"))
            .reset_index()
        )
        fig = px.bar(
            rating,
            x="Rating_Band",
            y="Average_Delivery_Time",
            title="Delivery Time by Partner Rating Band",
            color="Orders",
            color_continuous_scale=["#ffd23f", "#ff6b1a", "#ef3f23"],
        )
        fig.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, key="partner_rating_bar")


def page_live_dashboards(df: pd.DataFrame | None) -> None:
    hero("Live Interactive Dashboards", "Built from train.csv with live filters, cards, donut charts, bar charts, trend charts and decomposition-style analysis.")
    if df is None:
        st.warning(f"Dataset not found at: {TRAIN_PATH}")
        return
    filtered = filter_data(df)
    tab1, tab2, tab3 = st.tabs(["Delivery Overview", "Delay Analysis", "Partner & Vehicle"])
    with tab1:
        show_dashboard(filtered, "Delivery Overview Dashboard", "overview")
    with tab2:
        show_delay_dashboard(filtered)
    with tab3:
        show_partner_dashboard(filtered)


def manual_eta(record: dict) -> float:
    value = 18 + record["Delivery_distance_km"] * 1.7 + record["Pickup_delay_minutes"] * 0.45
    value += {"Low": 0, "Medium": 4, "High": 8, "Jam": 13}[record["Road_traffic_density"]]
    value += {"Sunny": 0, "Cloudy": 2, "Fog": 6, "Stormy": 7, "Windy": 4, "Sandstorms": 7}[record["Weatherconditions"]]
    value += record["multiple_deliveries"] * 3
    value += 5 if record["Festival"] == "Yes" else 0
    value -= max(record["Delivery_person_Ratings"] - 4.2, 0) * 2
    return max(value, 8)


def page_predictor() -> None:
    hero("Delivery Time Predictor", "Predict estimated delivery time and expected delay status using the Databricks-trained Bite Rush model.")
    model_bundle = load_model_bundle()
    if model_bundle is None:
        st.markdown(
            '<div class="callout">Trained model not found. Add bite_rush_best_model.pkl to this app folder to use model prediction.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="callout">Loaded model: {model_bundle.get("model_name", "Model")}</div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns(2)
    with left:
        age = st.number_input("Delivery Partner Age", 15, 60, 30)
        rating = st.number_input("Delivery Partner Rating", 1.0, 6.0, 4.6, step=0.1)
        vehicle_condition = st.selectbox("Vehicle Condition", [0, 1, 2, 3], index=2)
        multiple = st.selectbox("Multiple Deliveries", [0, 1, 2, 3], index=1)
        order_type = st.selectbox("Type of Order", ["Meal", "Snack", "Drinks", "Buffet"])
        vehicle_type = st.selectbox("Type of Vehicle", ["motorcycle", "scooter", "electric_scooter", "bicycle"])

    with right:
        restaurant_lat = st.number_input("Restaurant Latitude", value=12.971600, format="%.6f")
        restaurant_lon = st.number_input("Restaurant Longitude", value=77.594600, format="%.6f")
        delivery_lat = st.number_input("Delivery Latitude", value=12.981600, format="%.6f")
        delivery_lon = st.number_input("Delivery Longitude", value=77.604600, format="%.6f")
        weather = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Fog", "Stormy", "Windy", "Sandstorms"])
        traffic = st.selectbox("Traffic Density", ["Low", "Medium", "High", "Jam"])
        festival = st.selectbox("Festival", ["No", "Yes"])
        city = st.selectbox("City Type", ["Urban", "Metropolitian", "Semi-Urban"])

    c1, c2, c3 = st.columns(3)
    with c1:
        pickup_delay = st.number_input("Pickup Delay Minutes", 0, 180, 10)
    with c2:
        hour = st.slider("Order Hour", 0, 23, 18)
    with c3:
        day = st.selectbox("Order Day", [0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])

    dist = float(haversine_km(np.array([restaurant_lat]), np.array([restaurant_lon]), np.array([delivery_lat]), np.array([delivery_lon]))[0])
    record = {
        "Delivery_person_Age": age,
        "Delivery_person_Ratings": rating,
        "Vehicle_condition": vehicle_condition,
        "multiple_deliveries": multiple,
        "Delivery_distance_km": dist,
        "Pickup_delay_minutes": pickup_delay,
        "Order_hour": hour,
        "Order_day_of_week": day,
        "Weekend_flag": 1 if day in [5, 6] else 0,
        "Weatherconditions": weather,
        "Road_traffic_density": traffic,
        "Type_of_order": order_type,
        "Type_of_vehicle": vehicle_type,
        "Festival": festival,
        "City": city,
        "Rating_Band": "Low Rating" if rating < 4 else "Medium Rating" if rating < 4.7 else "High Rating",
        "Distance_Band": "Short Distance" if dist < 3 else "Medium Distance" if dist < 8 else "Long Distance",
    }

    if st.button("Predict Delivery Time", type="primary"):
        if model_bundle is not None and model_bundle.get("feature_cols") is not None:
            feature_cols = model_bundle["feature_cols"]
            input_df = pd.DataFrame([{col: record.get(col, np.nan) for col in feature_cols}])
            prediction = float(model_bundle["pipeline"].predict(input_df)[0])
            threshold = model_bundle.get("delay_threshold", DEFAULT_DELAY_THRESHOLD)
        else:
            prediction = manual_eta(record)
            threshold = DEFAULT_DELAY_THRESHOLD

        prediction = max(prediction, 1)
        status = "Delayed" if prediction > threshold else "On-Time"
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Estimated Delivery Time", f"{prediction:.1f} min")
        with c2:
            metric_card("Delay Status", status)
        with c3:
            metric_card("Distance", f"{dist:.2f} km")


def main() -> None:
    inject_css()
    df = load_data()
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=True)
        st.title("Bite Rush")
        page = st.radio("Pages", ["EDA Analysis", "Live Dashboards", "Delivery Time Predictor"])
        st.markdown("---")
        st.write("Built with Streamlit, Plotly, Databricks model output and Bite Rush delivery data.")

    if page == "EDA Analysis":
        page_eda(df)
    elif page == "Live Dashboards":
        page_live_dashboards(df)
    else:
        page_predictor()


if __name__ == "__main__":
    main()




