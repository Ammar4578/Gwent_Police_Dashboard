import io
import os
import sys
import json
import time
import math
import typing as t
from datetime import datetime

import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(
    page_title="Gwent Crime — Predictive Analytics Dashboard",
    page_icon="🚓",
    layout="wide"
)

st.title("Gwent Police Crime — Predictive Analytics Dashboard")
st.caption("EDA • Predictive Modeling • Interactive Insights")

@st.cache_data(show_spinner=False)
def load_csv(file: t.Union[str, io.BytesIO]) -> pd.DataFrame:
    """
    Load a Police.UK style street-level CSV.
    Tries to parse common column names. Adds parsed month and cleans types.
    """
    df = pd.read_csv(file)
    # Standardize column names (lower snake)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    if "month" in df.columns:
        try:
            df["month"] = pd.to_datetime(df["month"], errors="coerce")
        except Exception:
            pass

    # Ensure numeric for lat/lon if present
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Basic cleaning
    # Drop rows with no crime_type if it exists
    if "crime_type" in df.columns:
        df = df[~df["crime_type"].isna()]

    # Create year-month string for grouping
    if "month" in df.columns:
        df["year_month"] = df["month"].dt.to_period("M").astype(str)

    # Canonicalize location-like fields
    for col in ["lsoa_name", "lsoa_code", "location", "reported_by", "falls_within", "last_outcome_category"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


def pick_top_categories(series: pd.Series, top_n: int = 30) -> pd.Series:
    """
    Returns the same series but with rare categories grouped as 'Other'.
    """
    counts = series.value_counts(dropna=False)
    top = counts.head(top_n).index
    return series.where(series.isin(top), other="Other")


def basic_null_report(df: pd.DataFrame) -> pd.DataFrame:
    rep = pd.DataFrame({
        "column": df.columns,
        "non_null": df.notna().sum().values,
        "nulls": df.isna().sum().values,
        "pct_null": (df.isna().sum() / len(df) * 100).round(2).values
    })
    return rep


def make_confusion_df(y_true, y_pred, labels) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=pd.Index(labels, name="True"), columns=pd.Index(labels, name="Pred"))


# ==== Sidebar source selector ====
st.sidebar.header("📥 Data Source")
source = st.sidebar.radio(
    "Choose how to load data",
    ["Upload files", "Folder on server", "Upload a ZIP"],
    index=1  # default to Folder since that's what you want
)

drop_dupes = st.sidebar.toggle("Drop exact duplicate rows after merge", value=True)
show_nulls = st.sidebar.toggle("Show quick null report after merge", value=False)

# ---------- Common helpers ----------
from pathlib import Path
import zipfile
import pandas as pd

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (df.columns.astype(str).str.strip().str.lower()
                  .str.replace(" ", "_", regex=False).str.replace("-", "_", regex=False))
    return df

def _coerce_common_fields(df: pd.DataFrame) -> pd.DataFrame:
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        df["year_month"] = df["month"].dt.to_period("M").astype(str)
    for col in ("latitude", "longitude"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("lsoa_name", "lsoa_code", "location", "reported_by", "falls_within",
                "last_outcome_category", "crime_type"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

def _read_path_file(fp: Path) -> pd.DataFrame:
    suf = fp.suffix.lower()
    if suf == ".csv":
        df = pd.read_csv(fp, low_memory=False, encoding="utf-8-sig")
    elif suf in (".xlsx", ".xls"):
        # read all sheets if any, stack them
        xls = pd.ExcelFile(fp)
        frames = []
        for sh in xls.sheet_names:
            try:
                f = pd.read_excel(xls, sheet_name=sh)
                f["source_sheet"] = sh
                frames.append(f)
            except Exception:
                pass
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    elif suf == ".parquet":
        df = pd.read_parquet(fp)
    else:
        raise ValueError(f"Unsupported file: {fp}")
    if not df.empty:
        df = _normalize_columns(df)
        df["source_name"] = fp.name
    return df

def _merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = _coerce_common_fields(merged)
    return merged

# ---------- 1) Upload files (your existing flow) ----------
if source == "Upload files":
    multi_files = st.sidebar.file_uploader(
        "Upload multiple CSV / Excel / Parquet files",
        type=["csv", "xlsx", "xls", "parquet"],
        accept_multiple_files=True
    )

    @st.cache_data(show_spinner=False)
    def _read_uploaded(file) -> pd.DataFrame:
        name = file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(file, low_memory=False, encoding="utf-8-sig")
        elif name.endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(file)
            frames = []
            for sh in xls.sheet_names:
                try:
                    f = pd.read_excel(xls, sheet_name=sh)
                    f["source_sheet"] = sh
                    frames.append(f)
                except Exception:
                    pass
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        elif name.endswith(".parquet"):
            df = pd.read_parquet(file)
        else:
            df = pd.DataFrame()
        if not df.empty:
            df = _normalize_columns(df)
            df["source_name"] = file.name
        return df

    if multi_files:
        parts = []
        for f in multi_files:
            try:
                df_i = _read_uploaded(f)
                if not df_i.empty:
                    parts.append(df_i)
            except Exception as e:
                st.warning(f"Skipped {f.name}: {e}")
        if not parts:
            st.error("No readable data found in the uploaded files.")
            st.stop()
        df = _merge_frames(parts)
    else:
        st.info("👈 Upload multiple files to merge them here.")
        st.stop()

# ---------- 2) Folder on server (recursive) ----------
elif source == "Folder on server":
    st.sidebar.caption("Enter a folder path that the server can access.")
    root_path = st.sidebar.text_input("Folder path", value="", placeholder=r"e.g. D:\PoliceUK\Gwent or /home/you/data")
    patterns = st.sidebar.multiselect("File types to include", [".csv", ".xlsx", ".xls", ".parquet"],
                                      default=[".csv", ".xlsx", ".xls", ".parquet"])
    do_merge = st.sidebar.button("Scan & Merge")

    if not do_merge:
        st.stop()

    p = Path(root_path)
    if not p.exists() or not p.is_dir():
        st.error("That path doesn’t exist or isn’t a directory.")
        st.stop()

    files = [fp for fp in p.rglob("*") if fp.suffix.lower() in set(patterns)]
    if not files:
        st.error("No matching files found in that folder (including subfolders).")
        st.stop()

    frames, skipped = [], 0
    prog = st.progress(0)
    for i, fp in enumerate(files, 1):
        try:
            df_i = _read_path_file(fp)
            if not df_i.empty:
                frames.append(df_i)
        except Exception as e:
            skipped += 1
            st.warning(f"[Skipped] {fp} — {e}")
        finally:
            prog.progress(i / len(files))

    if not frames:
        st.error("No readable data found in the folder.")
        st.stop()

    df = _merge_frames(frames)

    if drop_dupes:
        before = len(df)
        df = df.drop_duplicates()
        deduped = before - len(df)
    else:
        deduped = 0

    st.success(f"Scanned **{len(files)}** files → merged **{len(frames)}** tables → "
               f"**{len(df):,}** rows • **{df.shape[1]}** columns"
               + (f" • removed **{deduped:,}** duplicates" if deduped else "")
               + (f" • skipped {skipped} file(s)" if skipped else ""))

# ---------- 3) Upload a ZIP of a folder ----------
else:  # "Upload a ZIP"
    zf = st.sidebar.file_uploader("Upload a .zip containing your folder", type=["zip"])
    if not zf:
        st.info("👈 Upload a ZIP file to merge everything inside it.")
        st.stop()

    # Read files inside ZIP without extracting to disk
    parts = []
    skipped = 0
    with zipfile.ZipFile(zf) as z:
        namelist = z.namelist()
        wanted = [n for n in namelist if Path(n).suffix.lower() in {".csv", ".xlsx", ".xls", ".parquet"}]
        if not wanted:
            st.error("No CSV/XLSX/Parquet files found inside the ZIP.")
            st.stop()

        prog = st.progress(0)
        for i, name in enumerate(wanted, 1):
            try:
                suf = Path(name).suffix.lower()
                with z.open(name) as f:
                    if suf == ".csv":
                        df_i = pd.read_csv(f, low_memory=False, encoding="utf-8-sig")
                    elif suf in (".xlsx", ".xls"):
                        # Need bytes for ExcelFile
                        data = f.read()
                        xls = pd.ExcelFile(io.BytesIO(data))
                        frames = []
                        for sh in xls.sheet_names:
                            try:
                                g = pd.read_excel(xls, sheet_name=sh)
                                g["source_sheet"] = sh
                                frames.append(g)
                            except Exception:
                                pass
                        df_i = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                    elif suf == ".parquet":
                        df_i = pd.read_parquet(io.BytesIO(f.read()))
                    else:
                        df_i = pd.DataFrame()
                if not df_i.empty:
                    df_i = _normalize_columns(df_i)
                    df_i["source_name"] = Path(name).name
                    parts.append(df_i)
            except Exception:
                skipped += 1
            finally:
                prog.progress(i / len(wanted))

    if not parts:
        st.error("Couldn’t read any tables from the ZIP.")
        st.stop()

    df = _merge_frames(parts)
    if drop_dupes:
        df = df.drop_duplicates()

    st.success(f"Merged **{len(parts)}** tables from ZIP → **{len(df):,}** rows • **{df.shape[1]}** columns"
               + (f" • skipped {skipped} items" if skipped else ""))

# ---------- Post-merge UI (shared) ----------
if show_nulls:
    rep = pd.DataFrame({
        "column": df.columns,
        "non_null": df.notna().sum().values,
        "nulls": df.isna().sum().values,
        "pct_null": (df.isna().sum() / len(df) * 100).round(2).values
    })
    st.dataframe(rep, use_container_width=True, height=320)

preview_cols = [c for c in ["month","year_month","crime_type","lsoa_name","location",
                            "last_outcome_category","latitude","longitude","source_name","source_sheet"]
                if c in df.columns]
st.dataframe(df[preview_cols].head(1000) if preview_cols else df.head(1000),
             use_container_width=True, height=380)

st.subheader("⬇️ Download merged data")
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download as CSV", data=csv_bytes, file_name="merged_data.csv", mime="text/csv")

try:
    import pyarrow  # parquet backend
    pq_path = "merged_data.parquet"
    df.to_parquet(pq_path, index=False)
    with open(pq_path, "rb") as f:
        st.download_button("Download as Parquet (recommended)", data=f.read(),
                           file_name="merged_data.parquet", mime="application/octet-stream")
except Exception:
    st.caption("Install `pyarrow` to enable Parquet export: `pip install pyarrow`")


st.sidebar.header("📥 Data")
st.sidebar.write("Upload your Police.UK CSV or use the sample path if running in this notebook.")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"], help="Police.UK street-level CSV (e.g., 2022-07-gwent-street.csv)")

# Optional sample path (you can replace with your local path)
default_path = "2022-07-gwent-street.csv"
use_sample = st.sidebar.toggle("Use example file name in current folder", value=False, help="If checked, the app will try to read '2022-07-gwent-street.csv' from the working directory.")

df = None
error_msg = None
if uploaded_file is not None:
    try:
        df = load_csv(uploaded_file)
    except Exception as e:
        error_msg = f"Failed to read uploaded file: {e}"
elif use_sample and os.path.exists(default_path):
    try:
        df = load_csv(default_path)
    except Exception as e:
        error_msg = f"Failed to read local sample '{default_path}': {e}"
else:
    st.info("👈 Upload a CSV in the sidebar to begin. You can download a sample Gwent file from Police.UK if needed.")
    st.stop()

if error_msg:
    st.error(error_msg)
    st.stop()

st.success(f"Loaded {len(df):,} rows • {df.shape[1]} columns")

with st.expander("🔎 Quick Null Report", expanded=False):
    st.dataframe(basic_null_report(df))

st.sidebar.header("🔍 Filters")

# Crime type filter
if "crime_type" in df.columns:
    crime_types = sorted(df["crime_type"].dropna().unique().tolist())
    selected_types = st.sidebar.multiselect("Crime types", crime_types, default=crime_types[: min(5, len(crime_types))])
else:
    selected_types = []

# Month range filter
if "month" in df.columns and pd.api.types.is_datetime64_any_dtype(df["month"]):
    min_date, max_date = df["month"].min(), df["month"].max()
    date_range = st.sidebar.date_input("Date range", value=(min_date.date(), max_date.date()), min_value=min_date.date(), max_value=max_date.date())
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = [pd.to_datetime(x) for x in date_range]
    else:
        start_date, end_date = min_date, max_date
else:
    start_date, end_date = None, None

# Geographic subset (LSOA) if available
if "lsoa_name" in df.columns:
    lsoas = sorted(df["lsoa_name"].dropna().unique().tolist())
    default_pick = lsoas[: min(10, len(lsoas))]
    selected_lsoas = st.sidebar.multiselect("LSOA(s)", lsoas, default=default_pick)
else:
    selected_lsoas = []

# Apply filters
df_f = df.copy()
if selected_types and "crime_type" in df_f.columns:
    df_f = df_f[df_f["crime_type"].isin(selected_types)]
if start_date is not None and end_date is not None and "month" in df_f.columns:
    df_f = df_f[(df_f["month"] >= start_date) & (df_f["month"] <= end_date)]
if selected_lsoas and "lsoa_name" in df_f.columns:
    df_f = df_f[df_f["lsoa_name"].isin(selected_lsoas)]

st.caption(f"Filters applied → Rows: **{len(df_f):,}**")


# EDA Design Starts from Here...............................

st.header("📊 Exploratory Data Analysis (EDA)")

colA, colB = st.columns(2)
with colA:
    if "crime_type" in df_f.columns:
        st.subheader("Crimes by Type")
        s = df_f["crime_type"].value_counts().reset_index()
        s.columns = ["crime_type", "count"]
        chart = alt.Chart(s).mark_bar().encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y("crime_type:N", sort="-x", title="Crime Type"),
            tooltip=["crime_type", "count"]
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Column 'crime_type' not found.")

with colB:
    if "year_month" in df_f.columns:
        st.subheader("Trend by Month")
        ts = df_f.groupby("year_month").size().reset_index(name="count")
        ts["year_month"] = pd.to_datetime(ts["year_month"])
        line = alt.Chart(ts).mark_line(point=True).encode(
            x=alt.X("year_month:T", title="Month"),
            y=alt.Y("count:Q", title="Crimes"),
            tooltip=["year_month:T", "count:Q"]
        )
        st.altair_chart(line, use_container_width=True)
    else:
        st.info("No 'month' column to build a timeline.")

colC, colD = st.columns(2)
with colC:
    if "lsoa_name" in df_f.columns:
        st.subheader("Top LSOAs")
        top_lsoa = df_f["lsoa_name"].value_counts().head(20).reset_index()
        top_lsoa.columns = ["lsoa_name", "count"]
        bar = alt.Chart(top_lsoa).mark_bar().encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y("lsoa_name:N", sort="-x", title="LSOA"),
            tooltip=["lsoa_name", "count"]
        )
        st.altair_chart(bar, use_container_width=True)
    else:
        st.info("Column 'lsoa_name' not found.")

with colD:
    if {"latitude", "longitude"}.issubset(df_f.columns):
        st.subheader("Crime Map (sample up to 5,000 points)")
        map_df = df_f[["latitude", "longitude"]].dropna().sample(min(5000, len(df_f)), random_state=42) if len(df_f) > 0 else df_f
        st.map(map_df.rename(columns={"latitude":"lat", "longitude":"lon"}))
    else:
        st.info("Latitude/Longitude not available for map.")

with st.expander("🔬 Drill-down Table", expanded=False):
    cols_show = [c for c in ["month", "crime_type", "lsoa_name", "location", "last_outcome_category", "latitude", "longitude"] if c in df_f.columns]
    st.dataframe(df_f[cols_show].head(1000))

# Predictive Modeling..............................

st.header("Predictive Model")

# Target selection
possible_targets = [c for c in ["crime_type", "last_outcome_category"] if c in df.columns]
if not possible_targets:
    st.warning("No suitable target column found (expected 'crime_type' or 'last_outcome_category').")
    st.stop()

target_col = st.selectbox("Choose target to predict", options=possible_targets, index=0, help="Choose the dependent variable.")

# Feature candidates
candidate_features = [c for c in ["lsoa_name", "location", "reported_by", "falls_within", "year_month"] if c in df.columns and c != target_col]
if {"latitude", "longitude"}.issubset(df.columns):
    candidate_features += ["latitude", "longitude"]

st.caption("Tip: Too many categorical levels can hurt performance. We'll auto-bucket rare categories into 'Other'.")

# Prepare modeling dataframe
model_df = df.dropna(subset=[target_col]).copy()

# Rare category bucketing for high-cardinality categoricals
for col in candidate_features:
    if model_df[col].dtype == "object":
        model_df[col] = pick_top_categories(model_df[col], top_n=40)

# Train/test split controls
test_size = st.slider("Test size (holdout %)", 10, 40, 20, step=5) / 100.0
random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

# Selected features UI
selected_features = st.multiselect("Select features", options=candidate_features, default=candidate_features[:5])

if not selected_features:
    st.warning("Select at least one feature to train the model.")
    st.stop()

# Build X, y
X = model_df[selected_features].copy()
y = model_df[target_col].astype(str)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None)

# Preprocess: categorical vs numeric
cat_cols = [c for c in selected_features if X[c].dtype == "object"]
num_cols = [c for c in selected_features if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(with_mean=False), num_cols)
    ]
)

# Model choice
model_choice = st.selectbox("Model", ["Logistic Regression", "Random Forest"], index=1)

if model_choice == "Logistic Regression":
    clf = LogisticRegression(max_iter=1000, n_jobs=None if hasattr(LogisticRegression(), "n_jobs") else None)
else:
    clf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)

pipe = Pipeline(steps=[("prep", preprocess), ("model", clf)])

# Train
with st.spinner("Training model..."):
    pipe.fit(X_train, y_train)

# Predict & metrics
y_pred = pipe.predict(X_test)
labels = sorted(y.unique().tolist())
acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")

mcol1, mcol2, mcol3 = st.columns(3)
mcol1.metric("Accuracy", f"{acc:.3f}")
mcol2.metric("Macro F1", f"{f1m:.3f}")
mcol3.metric("Classes", f"{len(labels)}")

st.subheader("Confusion Matrix")
cm_df = make_confusion_df(y_test, y_pred, labels)
cm_chart = px.imshow(cm_df.values, x=labels, y=labels, labels=dict(x="Predicted", y="True", color="Count"))
st.plotly_chart(cm_chart, use_container_width=True)

with st.expander("📄 Classification Report", expanded=False):
    st.text(classification_report(y_test, y_pred, zero_division=0))

# -------------------------
# Interactive Predictor
# -------------------------
st.header("Try a Prediction")

def make_predict_input_ui(selected_features: t.List[str]) -> pd.DataFrame:
    """
    Renders input widgets for each selected feature and returns a single-row DataFrame.
    """
    inputs = {}
    cols = st.columns(min(3, len(selected_features)) if selected_features else 1)
    for i, feat in enumerate(selected_features):
        with cols[i % len(cols)]:
            if feat in cat_cols:
                # Offer choices from training data (post-bucketing)
                choices = sorted(X_train[feat].astype(str).unique().tolist())
                val = st.selectbox(f"{feat}", options=choices, index=min(0, len(choices)-1) if choices else 0)
                inputs[feat] = val
            else:
                # numeric
                s = X_train[feat].dropna()
                default = float(s.median() if len(s) else 0.0)
                minv = float(s.min() if len(s) else 0.0)
                maxv = float(s.max() if len(s) else 1.0)
                val = st.number_input(f"{feat}", value=default)
                inputs[feat] = val
    return pd.DataFrame([inputs])

pred_input = make_predict_input_ui(selected_features)

if st.button("Predict"):
    pred = pipe.predict(pred_input)[0]
    st.success(f"**Predicted {target_col}:** {pred}")

st.header("📥 Downloads")

# Filtered data
csv_bytes = df_f.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered data (CSV)", data=csv_bytes, file_name="gwent_filtered.csv", mime="text/csv")

# Trained model (pickle)
try:
    import pickle
    model_bytes = pickle.dumps(pipe)
    st.download_button("Download trained model (pickle)", data=model_bytes, file_name="gwent_model.pkl", mime="application/octet-stream")
except Exception as e:
    st.info(f"Pickle unavailable: {e}")


st.header("Methodology & Business Recommendations")
st.markdown("")

st.caption("Built by Ammar Ahmed Roll no 100496689 using Python on Streamlit")

