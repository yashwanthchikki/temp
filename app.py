import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -----------------------------
# Sample Data (replace with your real data load)
# -----------------------------
np.random.seed(42)
data = pd.DataFrame({
    "Advisor": [f"Advisor {i}" for i in range(1, 21)],
    "MD_ratio": np.random.randint(1, 5, 20),
    "Client_count": np.random.randint(10, 100, 20),
    "Clients_individual": np.random.randint(5, 40, 20),
    "Clients_team": np.random.randint(5, 40, 20),
    "Total_AUM": np.random.randint(1e6, 5e6, 20),
    "External_assets": np.random.randint(1e6, 5e6, 20),
    "Last_onboarded_date": pd.date_range("2024-01-01", periods=20, freq="30D"),
    "Recency_days": np.random.randint(1, 400, 20),
    "Frequency_2024": np.random.randint(1, 50, 20),
    "Frequency_2025": np.random.randint(1, 50, 20),
    "Products_per_client": np.round(np.random.uniform(1, 5, 20), 2)
})

# Derived columns
data["Total_assets"] = data["Total_AUM"] + data["External_assets"]
data["Pct_external"] = data["External_assets"] / data["Total_assets"]
data["Growth_rate"] = data["Frequency_2025"] / (data["Frequency_2024"] + 1)
data["Ind_ratio"] = data["Clients_individual"] / (data["Client_count"] + 1)
data["Team_ratio"] = data["Clients_team"] / (data["Client_count"] + 1)
data["AUM_per_client"] = data["Total_AUM"] / (data["Client_count"] + 1)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Ranking Controls")

# Filter: last onboarded date
min_date, max_date = data["Last_onboarded_date"].min(), data["Last_onboarded_date"].max()
date_filter = st.sidebar.date_input("Filter by onboarded date range", [min_date, max_date])

# Core weights
core_weights = {
    "MD_ratio": st.sidebar.slider("Weight: MD Ratio", 0.0, 1.0, 0.1),
    "Client_count": st.sidebar.slider("Weight: Client Count", 0.0, 1.0, 0.1),
    "Clients_individual": st.sidebar.slider("Weight: Clients Individual", 0.0, 1.0, 0.1),
    "Clients_team": st.sidebar.slider("Weight: Clients Team", 0.0, 1.0, 0.1),
    "Total_AUM": st.sidebar.slider("Weight: Total AUM", 0.0, 1.0, 0.2),
    "External_assets": st.sidebar.slider("Weight: External Assets", 0.0, 1.0, 0.1),
    "Recency_days": st.sidebar.slider("Weight: Recency", 0.0, 1.0, 0.1),
    "Frequency_2024": st.sidebar.slider("Weight: New Clients 2024", 0.0, 1.0, 0.05),
    "Frequency_2025": st.sidebar.slider("Weight: New Clients 2025", 0.0, 1.0, 0.05),
    "Products_per_client": st.sidebar.slider("Weight: Products per Client", 0.0, 1.0, 0.1),
}

# Normalize weights
total_wt = sum(core_weights.values())
if total_wt == 0: total_wt = 1
norm_weights = {k: v / total_wt for k, v in core_weights.items()}

# -----------------------------
# Ranking calculation
# -----------------------------
rank_df = data.copy()
rank_df["Part_of_team_flag"] = np.random.randint(0, 2, len(rank_df))

# Toggle filter in sidebar
st.sidebar.header("Extra Filters")
toggle_val = st.sidebar.radio("concierge_flag", ("All", "1 only", "0 only"))

if toggle_val == "1 only":
    rank_df = rank_df[rank_df["Part_of_team_flag"] == 1]
elif toggle_val == "0 only":
    rank_df = rank_df[rank_df["Part_of_team_flag"] == 0]

def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-6)

for col in core_weights.keys():
    if col == "Recency_days":
        rank_df[col + "_norm"] = 1 - normalize(rank_df[col])
    else:
        rank_df[col + "_norm"] = normalize(rank_df[col])

rank_df["Score"] = sum(rank_df[col + "_norm"] * w for col, w in norm_weights.items())

# Apply filter by last onboarded date
rank_df = rank_df[
    (rank_df["Last_onboarded_date"] >= pd.to_datetime(date_filter[0])) &
    (rank_df["Last_onboarded_date"] <= pd.to_datetime(date_filter[1]))
]

# -----------------------------
# Extra Options: Manual AUM filter
# -----------------------------

# -----------------------------
# Portfolio Summary
# -----------------------------
st.title("Advisor Ranking Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Total Advisors", len(rank_df))
col2.metric("Total Clients", int(rank_df["Client_count"].sum()))
col3.metric("Total Assets (M)", f"${rank_df['Total_assets'].sum()/1e6:.1f}M")

# -----------------------------
# Portfolio Plots (Improved Design)
# -----------------------------
# -----------------------------
# Portfolio Plots (Distribution)
# -----------------------------
st.subheader("Portfolio Insights")

# Distribution of AUM
fig, ax = plt.subplots(figsize=(7,4))
ax.hist(rank_df["Total_AUM"]/1e6, bins=10, color="#4C72B0", alpha=0.8, edgecolor='black')
ax.set_title("Distribution of AUM Across Advisors (in $M)")
ax.set_xlabel("AUM ($M)")
ax.set_ylabel("Number of Advisors")
st.pyplot(fig)

# Distribution of Recency
fig, ax = plt.subplots(figsize=(7,4))
ax.hist(rank_df["Recency_days"], bins=10, color="#55A868", alpha=0.8, edgecolor='black')
ax.set_title("Distribution of Recency Across Advisors (Days)")
ax.set_xlabel("Days since last onboarding")
ax.set_ylabel("Number of Advisors")
st.pyplot(fig)


# -----------------------------
# Trend Plots (Monthly Aggregation)
# -----------------------------
trend_df = rank_df.copy()

clients_trend_month = trend_df.groupby(trend_df["Last_onboarded_date"].dt.to_period("M"))["Client_count"].sum()
clients_trend_month.index = clients_trend_month.index.to_timestamp()

aum_trend_month = trend_df.groupby(trend_df["Last_onboarded_date"].dt.to_period("M"))["Total_AUM"].sum()
aum_trend_month.index = aum_trend_month.index.to_timestamp()

# Client Onboarding Trend
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(clients_trend_month.index, clients_trend_month.values, marker="o", linestyle='-', color="#FF6F61")
ax.set_title("Monthly Client Onboarding Trend")
ax.set_xlabel("Month")
ax.set_ylabel("Clients Onboarded")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# AUM Trend
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(aum_trend_month.index, aum_trend_month.values/1e6, marker="o", linestyle='-', color="#6A5ACD")
ax.set_title("Monthly Total AUM Trend")
ax.set_xlabel("Month")
ax.set_ylabel("Total AUM ($M)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# -----------------------------
# Ranked Advisors Table
# -----------------------------
st.subheader("Ranked Advisors")
st.dataframe(rank_df[["Advisor", "Score", "Client_count", "Total_AUM", "External_assets", "Recency_days"]])

# -----------------------------
# Advisor Details
# -----------------------------
st.subheader("Advisor Details")
selected = st.selectbox("Select an advisor", rank_df["Advisor"])

if selected:
    adv = rank_df[rank_df["Advisor"] == selected].iloc[0]
    st.write(f"### {adv['Advisor']} - Score: {adv['Score']:.2f}")

    col1, col2 = st.columns(2)

    # Pie: Assets split
    with col1:
        fig, ax = plt.subplots()
        ax.pie([adv["Total_AUM"], adv["External_assets"]],
               labels=["Our AUM", "External"],
               autopct="%1.1f%%", startangle=90, colors=["#4C72B0","#55A868"])
        ax.set_title("Assets Split")
        st.pyplot(fig)

    # Pie: Clients split
    with col2:
        fig, ax = plt.subplots()
        ax.pie([adv["Clients_individual"], adv["Clients_team"]],
               labels=["Individual", "Team"],
               autopct="%1.1f%%", startangle=90, colors=["#FF6F61","#6A5ACD"])
        ax.set_title("Client Split")
        st.pyplot(fig)

    st.write("**Additional Metrics:**")
    st.write(f"- Growth Rate (2025/2024): {adv['Growth_rate']:.2f}")
    st.write(f"- % External Assets: {adv['Pct_external']:.2%}")
    st.write(f"- AUM per Client: ${adv['AUM_per_client']:.0f}")
    st.write(f"- MD Ratio: {adv['MD_ratio']}")


with st.expander("Advanced Metrics & Filters"):
    st.write("Adjust advanced metrics if managers want:")
    adv_growth_weight = st.slider("Weight: Growth Rate", 0.0, 1.0, 0.0)
    adv_ext_ratio_weight = st.slider("Weight: % External Assets", 0.0, 1.0, 0.0)

    st.write("Advanced filters:")
    aum_min_input = st.number_input("Min AUM ($)", min_value=0, value=0, step=100000)
    aum_max_input = st.number_input("Max AUM ($)", min_value=0, value=int(rank_df["Total_AUM"].max()), step=100000)

    # Apply manual AUM filter
    rank_df = rank_df[
        (rank_df["Total_AUM"] >= aum_min_input) & 
        (rank_df["Total_AUM"] <= aum_max_input)
    ]
