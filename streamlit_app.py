import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
from datetime import datetime


st.set_page_config(page_title="OARI — Operator Dashboard", layout="wide")

# ---------------- Mock data ----------------
hours = [f"{h}:00" for h in range(24)]
timeline = pd.DataFrame({
    "hour": hours,
    "risk": np.round((30 + 60 * np.abs(np.sin(np.arange(24) / 3))) / 100, 2),
    "incidents": np.round(2 + 6 * np.abs(np.cos(np.arange(24) / 4))).astype(int),
})

runs = pd.DataFrame([
    {"name": "The Chute",   "type": "Double Black", "icon": "♦♦", "risk": 0.82, "staff": 3, "status": "Open"},
    {"name": "Western Sun", "type": "Black",        "icon": "♦",  "risk": 0.61, "staff": 2, "status": "Advisory"},
    {"name": "Jack Rabbit", "type": "Blue",         "icon": "■",  "risk": 0.46, "staff": 1, "status": "Open"},
    {"name": "Tree Line",   "type": "Green",        "icon": "●",  "risk": 0.18, "staff": 1, "status": "Open"},
])

variables = [
    {"key": "rain",       "label": "Rain (mm/hr)",     "min": 0.0,  "max": 10.0,  "value": 1.2},
    {"key": "temp",       "label": "Air Temp (°C)",    "min": -25., "max": 10.0,  "value": -6.0},
    {"key": "uv",         "label": "UV Index",         "min": 0.0,  "max": 11.0,  "value": 2.0},
    {"key": "wind",       "label": "Wind (km/h)",      "min": 0.0,  "max": 80.0,  "value": 24.0},
    {"key": "humidity",   "label": "Humidity (%)",     "min": 10.0, "max": 100.0, "value": 68.0},
    {"key": "cloud",      "label": "Cloud Cover (%)",  "min": 0.0,  "max": 100.0, "value": 55.0},
    {"key": "snowDepth",  "label": "Snow Depth (cm)",  "min": 0.0,  "max": 300.0, "value": 112.0},
    {"key": "visibility", "label": "Visibility (m)",   "min": 50.0, "max": 3000., "value": 900.0},
]

# ---------------- Sidebar controls ----------------
st.sidebar.title("Scenario Controls")
# If segmented_control isn't available in your Streamlit version, fall back to selectbox:
try:
    mode = st.sidebar.segmented_control("Model Mode", ["In-house", "Open"], selection_mode="single")
except Exception:
    mode = st.sidebar.selectbox("Model Mode", ["In-house", "Open"])
hazard = st.sidebar.selectbox("Target Hazard", ["Slip / Fall", "Collision (tree/person)", "Hypothermia", "Equipment Failure"])

st.sidebar.write("### Variables")
vals = {}
for v in variables:
    vals[v["key"]] = st.sidebar.slider(
        v["label"], min_value=float(v["min"]), max_value=float(v["max"]),
        value=float(v["value"])
    )

# ---------------- Simple risk function (placeholder) ----------------
def headline_risk(val_dict):
    norm = []
    for v in variables:
        key = v["key"]
        n = (val_dict[key] - v["min"]) / (v["max"] - v["min"])
        norm.append(n)
    avg = np.mean(norm)
    return float(np.clip(0.2 + 0.6 * avg, 0.02, 0.98))

R = headline_risk(vals)

# ---------------- Header/KPIs ----------------
left, right = st.columns([1, 1.8])
with left:
    st.markdown("### OARI — Operator Dashboard")
    st.caption(f"Mode: **{mode}** • Hazard: **{hazard}**")
with right:
    k1, k2, _ = st.columns([1,1,1])
    k1.metric("Current Risk", f"{int(round(R*100))}%")
    k2.metric("Predicted Incidents (next 24h)", int(round(R*18)))

st.divider()

# ---------------- Charts ----------------
c1, c2, c3 = st.columns(3)

scaled = timeline.copy()
scaled["adj_risk"] = np.clip(scaled["risk"] * (R / 0.8), 0, 1)

risk_chart = (
    alt.Chart(scaled).mark_area(opacity=0.4)
    .encode(x="hour", y=alt.Y("adj_risk", title="Risk (0–1)"))
    .properties(height=260)
)
inc_chart = (
    alt.Chart(timeline).mark_bar()
    .encode(x="hour", y=alt.Y("incidents", title="Incidents"))
    .properties(height=260)
)
pr_df = pd.DataFrame({"precision":[1,0.8,0.6,0.4,0.2], "recall":[0.2,0.4,0.55,0.68,0.8]})
pr_chart = (
    alt.Chart(pr_df).mark_line(point=True)
    .encode(x=alt.X("precision", title="Precision"), y=alt.Y("recall", title="Recall"))
    .properties(height=260)
)

c1.subheader("Risk Timeline"); c1.altair_chart(risk_chart, use_container_width=True)
c2.subheader("Incidents");     c2.altair_chart(inc_chart, use_container_width=True)
c3.subheader("Precision-Recall"); c3.altair_chart(pr_chart, use_container_width=True)

st.divider()

# ---------------- Runs & allocation (live, stable) ----------------
st.subheader("Runs & Allocation")

# One-time init
if "total_staff" not in st.session_state:
    st.session_state.total_staff = 10
if "staff_assign" not in st.session_state:
    st.session_state.staff_assign = {row["name"]: int(row["staff"]) for _, row in runs.iterrows()}
for _, row in runs.iterrows():
    name = row["name"]
    st.session_state.staff_assign.setdefault(name, int(row["staff"]))
    st.session_state.setdefault(f"staff_input_{name}", st.session_state.staff_assign[name])

def _cap_overflow():
    pool = int(st.session_state.total_staff)
    assign = st.session_state.staff_assign
    total = sum(assign.values())
    if total <= pool or total == 0:
        # sync widgets to assignment (keeps things consistent)
        for n, v in assign.items():
            st.session_state[f"staff_input_{n}"] = int(v)
        return

    # Proportional scale + round, then distribute leftover
    scale = pool / total
    updated = {n: int(np.floor(v * scale)) for n, v in assign.items()}
    leftover = pool - sum(updated.values())
    fracs = sorted(((n, assign[n]*scale - np.floor(assign[n]*scale)) for n in assign),
                   key=lambda x: x[1], reverse=True)
    for i in range(leftover):
        updated[fracs[i % len(fracs)][0]] += 1

    # write back (both canonical + widgets)
    for n, v in updated.items():
        st.session_state.staff_assign[n] = int(v)
        st.session_state[f"staff_input_{n}"] = int(v)

def on_total_change():
    _cap_overflow()

def on_run_change(name: str):
    # pull widget value → canonical state
    new_val = int(st.session_state.get(f"staff_input_{name}", 0))
    st.session_state.staff_assign[name] = new_val
    _cap_overflow()

# Controls
st.number_input(
    "Total available staff",
    min_value=0, max_value=1000, step=1,
    key="total_staff",
    on_change=on_total_change
)

query = st.text_input("Search runs…", "")
view = runs[runs["name"].str.contains(query, case=False, na=False)].copy()

def bucket(x):
    return "High" if x >= 0.75 else ("Moderate" if x >= 0.45 else "Low")

pool = int(st.session_state.total_staff)
current_total = sum(st.session_state.staff_assign.values())

for _, row in view.iterrows():
    name = row["name"]
    cur = int(st.session_state.staff_assign.get(name, 0))

    # capacity left if this run stays at 'cur'
    remaining_excl_this = pool - (current_total - cur)
    # IMPORTANT: never let max drop below current (prevents Streamlit auto-clamping)
    max_for_run = max(cur, cur + remaining_excl_this)

    a, b, c = st.columns([2, 1, 1])
    with a:
        st.markdown(f"**{name}**  \n<small>{row['icon']} {row['type']}</small>", unsafe_allow_html=True)
    with b:
        st.metric("Risk", f"{int(row['risk']*100)}%", bucket(row["risk"]))
    with c:
        st.number_input(
            f"Staff — {name}",
            min_value=0,
            max_value=int(max_for_run),
            step=1,
            key=f"staff_input_{name}",
            on_change=on_run_change,
            args=(name,)  # <-- use args, not kwargs
        )

assigned = sum(st.session_state.staff_assign.values())
remaining = pool - assigned
if remaining < 0:
    st.error(f"Over-assigned by {-remaining} staff.")
elif remaining == 0:
    st.info("All staff assigned.")
else:
    st.success(f"Assigned {assigned}. Remaining: {remaining}.")


# ---------------- Upload hook (for later) ----------------
with st.expander("Optional: Upload a CSV (placeholder for your data)"):
    f = st.file_uploader("Choose a CSV", type=["csv"])
    if f is not None:
        df = pd.read_csv(f)
        st.write(df.head())

# ==================== Data Upload & Template Tab ====================
st.divider()
st.subheader("Data & Upload")

tab_dashboard, tab_upload = st.tabs(["📊 Dashboard (current)", "📥 Upload & Template"])

# -- Template definition --
TEMPLATE_COLUMNS = [
    "Observation #", "Date", "Time", "Injured (Y/N)", "Rain (mm/hr)", "Air Temp (°C)",
    "UV Index", "Wind (km/h)", "Humidity (%)", "Cloud Cover (%)",
    "Snow Depth (cm)", "Visibility (m)", "Ski Run"
]

def make_template_df() -> pd.DataFrame:
    demo = pd.DataFrame({
        "Observation #": [1, 2, 3],
        "Date": ["2025-01-20", "2025-01-20", "2025-01-21"],
        "Time": ["09:30", "13:15", "15:45"],
        "Injured (Y/N)": ["N", "Y", "N"],
        "Rain (mm/hr)": [0.0, 0.6, 0.0],
        "Air Temp (°C)": [-8, -5, -3],
        "UV Index": [1, 2, 3],
        "Wind (km/h)": [12, 30, 18],
        "Humidity (%)": [70, 65, 60],
        "Cloud Cover (%)": [40, 90, 20],
        "Snow Depth (cm)": [120, 120, 121],
        "Visibility (m)": [1000, 600, 1500],
        "Ski Run": ["The Chute", "Western Sun", "Jack Rabbit"],
    })
    # Ensure column order
    return demo[TEMPLATE_COLUMNS]

def template_bytes() -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        make_template_df().to_excel(buf, index=False, sheet_name="OARI_Data")
    buf.seek(0)
    return buf.getvalue()

with tab_upload:
    st.markdown("### Upload observations or download the Excel template")

    # Download button
    st.download_button(
        label="⬇️ Download Excel template",
        data=template_bytes(),
        file_name="OARI_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Download a blank template with the required columns."
    )

    st.write("— or —")

    # Upload widget
    up = st.file_uploader(
        "Upload your Excel file (*.xlsx) with the exact headers",
        type=["xlsx"],
        accept_multiple_files=False
    )

    if up is not None:
        try:
            df_raw = pd.read_excel(up, engine="openpyxl")
        except Exception as e:
            st.error(f"Could not read Excel file: {e}")
            st.stop()

        # Validate columns
        missing = [c for c in TEMPLATE_COLUMNS if c not in df_raw.columns]
        if missing:
            st.error(f"Missing required column(s): {', '.join(missing)}")
            st.info("Tip: download the template above to match exact headers.")
            st.dataframe(df_raw.head())
            st.stop()

        # Reorder to template order and make a working copy
        df = df_raw[TEMPLATE_COLUMNS].copy()

        # Parse / clean
        # 1) Date & Time → combine to a single datetime (where possible)
        def to_time(s):
            try:
                return datetime.strptime(str(s).strip(), "%H:%M").time()
            except Exception:
                return None

        def to_date(s):
            # Allow already-parsed dates or strings like YYYY-MM-DD
            if isinstance(s, datetime):
                return s.date()
            try:
                return pd.to_datetime(str(s)).date()
            except Exception:
                return None

        df["Date_parsed"] = df["Date"].apply(to_date)
        df["Time_parsed"] = df["Time"].apply(to_time)
        df["DateTime"] = pd.to_datetime(
            df["Date_parsed"].astype(str) + " " + df["Time_parsed"].astype(str),
            errors="coerce"
        )

        # 2) Injured → boolean
        def to_bool(y):
            if str(y).strip().upper() in ("Y", "YES", "1", "TRUE"):
                return True
            if str(y).strip().upper() in ("N", "NO", "0", "FALSE"):
                return False
            return None

        df["Injured"] = df["Injured (Y/N)"].apply(to_bool)

        # 3) Numeric columns (coerce errors to NaN)
        numeric_cols = [
            "Rain (mm/hr)", "Air Temp (°C)", "UV Index", "Wind (km/h)",
            "Humidity (%)", "Cloud Cover (%)", "Snow Depth (cm)", "Visibility (m)"
        ]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # --- Display ---
        st.success("File uploaded and parsed successfully.")
        st.markdown("#### Preview")
        st.dataframe(df.head(20), use_container_width=True)

        # --- Quick interpretation widgets ---
        st.markdown("#### Summary")
        total_obs = len(df)
        injuries = int(df["Injured"].fillna(False).sum())
        injury_rate = (injuries / total_obs * 100) if total_obs else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Total observations", total_obs)
        c2.metric("Injuries", injuries)
        c3.metric("Injury rate", f"{injury_rate:.1f}%")

        # Injuries by Ski Run
        st.markdown("#### Injuries by Ski Run")
        by_run = (
            df.groupby("Ski Run")
              .agg(
                  observations=("Ski Run", "size"),
                  injuries=("Injured", lambda s: int(pd.Series(s).fillna(False).sum()))
              )
              .reset_index()
        )
        by_run["injury_rate_%"] = (by_run["injuries"] / by_run["observations"] * 100).round(1)

        # Bar chart (Altair)
        chart = (
            alt.Chart(by_run)
            .mark_bar()
            .encode(
                x=alt.X("Ski Run:N", sort="-y", title="Ski Run"),
                y=alt.Y("injuries:Q", title="Injuries"),
                tooltip=["Ski Run", "observations", "injuries", "injury_rate_%"]
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

        # Descriptive statistics for numeric drivers
        st.markdown("#### Conditions (descriptive stats)")
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

        # Time-of-day distribution (if datetimes were parsed)
        if df["DateTime"].notna().any():
            st.markdown("#### Time-of-day distribution")
            tmp = df[df["DateTime"].notna()].copy()
            tmp["hour"] = tmp["DateTime"].dt.hour
            hist = (
                alt.Chart(tmp)
                .mark_bar()
                .encode(x=alt.X("hour:O", title="Hour of day"), y="count()")
                .properties(height=220)
            )
            st.altair_chart(hist, use_container_width=True)
        else:
            st.info("Time values could not be parsed in some rows—check the Time column format (HH:MM).")
