import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

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

# ---------------- Runs & allocation ----------------
st.subheader("Runs & Allocation")
query = st.text_input("Search runs…", "")
view = runs[runs["name"].str.contains(query, case=False, na=False)].copy()

def bucket(x): 
    return "High" if x>=0.75 else ("Moderate" if x>=0.45 else "Low")

for idx, row in view.iterrows():
    a,b,c = st.columns([2,1,1])
    with a:
        st.markdown(f"**{row['name']}**  \n<small>{row['icon']} {row['type']}</small>", unsafe_allow_html=True)
    with b:
        st.metric("Risk", f"{int(row['risk']*100)}%", bucket(row["risk"]))
    with c:
        new_staff = st.number_input(f"Staff — {row['name']}", min_value=0, max_value=20, value=int(row["staff"]), key=f"staff_{idx}")
        view.loc[idx, "staff"] = new_staff

st.caption("Tip: these are demo numbers. Hook your model later to replace the risk function and charts.")

# ---------------- Upload hook (for later) ----------------
with st.expander("Optional: Upload a CSV (placeholder for your data)"):
    f = st.file_uploader("Choose a CSV", type=["csv"])
    if f is not None:
        df = pd.read_csv(f)
        st.write(df.head())
