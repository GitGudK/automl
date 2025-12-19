"""
EDA for dependent variables in the sepsis dataset.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def print_analysis_note(note: str):
    print("================================================================================")
    print(f"ANALYSIS: {note}")
    print("================================================================================")


BASE_PATH = os.path.dirname(os.path.dirname(__file__))

# Basic info
df_raw_data = pd.read_csv(BASE_PATH + "/data/data_science_project_data.csv")
col_names = sorted([i for i in df_raw_data.columns])
df_raw_data.info()
df_raw_data.describe()
print(df_raw_data.head())

# Time range
df_raw_data["timestamp"] = pd.to_datetime(df_raw_data["timestamp"])
min_date = df_raw_data["timestamp"].min()
max_date = df_raw_data["timestamp"].max()
print(f"Data ranges from {min_date} to {max_date}")

# Initialize some analysis notes for aggregation later
analysis_notes = []

# DEPENDENT VARIABLE ANALYSIS
# ===============================================================================
# Average time resolution of the dvar data (just rough estimate, not accounting for visits)
avg_time_diffs = []
df_data_sorted_by_date = df_raw_data.sort_values(by="timestamp", ascending=True)
df_data_sorted_by_date = df_data_sorted_by_date.reset_index(drop=True)
obs_dates = df_data_sorted_by_date["timestamp"].drop_duplicates().sort_values().reset_index(drop=True)
time_diffs = obs_dates.diff().dropna()
time_diffs = pd.Series([t for t in time_diffs if t <= pd.Timedelta(hours=6)]) # exclude large gaps
avg_time_diff = time_diffs.mean()
avg_time_diffs.append(avg_time_diff)
overall_avg_time_diff = sum(avg_time_diffs, pd.Timedelta(0)) / len(avg_time_diffs)
print(f"Overall average time difference between observations: {overall_avg_time_diff}")
note = "average time difference between dvar observations is roughly 20 seconds"
analysis_notes.append(note)
print_analysis_note(note)

# Client ids
unique_entitys = sorted(df_raw_data["entity"].unique())
print(f"Unique entitys: {unique_entitys}")
note = f"{len(unique_entitys)} unique entitys in the dataset, assume these are population groups"
analysis_notes.append(note)
print_analysis_note(note)

# Visit ids
unique_session_ids = sorted(df_raw_data["session_id"].unique())
print(f"Unique visit IDs: {unique_session_ids}")
note = f"{len(unique_session_ids)} unique visit IDs in the dataset, assume these are individual patients"
analysis_notes.append(note)
print_analysis_note(note)

# Dependent variable for a single entity with patient (session_id) for 10 days
sample_entity = unique_entitys[0]
filt = (df_raw_data["entity"] == sample_entity) & (df_raw_data["timestamp"] >= min_date + pd.Timedelta(days=10)) & \
    (df_raw_data["timestamp"] <= min_date + pd.Timedelta(days=20))
entity_data_filt = df_raw_data.loc[filt, :]
entity_data_filt = entity_data_filt.sort_values(by="timestamp", ascending=True)
visit_condition_count = entity_data_filt.groupby("session_id")["flag_positive"].count().reset_index()
visit_condition_sum = entity_data_filt.groupby("session_id")["flag_positive"].sum().reset_index()
visit_condition_mean = entity_data_filt.groupby("session_id")["flag_positive"].mean().reset_index()
note = "number of timestamps vary significantly per session_id / patient"
analysis_notes.append(note)
print_analysis_note(note)

plt.figure(figsize=(10, 5))
plt.plot(visit_condition_mean["session_id"], visit_condition_mean["flag_positive"], marker='o',
            linestyle='-', color='blue')
plt.xlabel("Visit ID")
plt.ylabel("Condition Present Ratio per Visit")
plt.title(f"Condition Present Ratio per Visit for Client {sample_entity}")
plt.ylim(-0.1, 1.1)
plt.tight_layout()
plt.show()

# Check if flag_positive ever changes within a single session_id
visits_with_changes = df_raw_data.groupby("session_id")["flag_positive"].nunique()
visits_with_changes = visits_with_changes[visits_with_changes > 1]
print(f"Number of visit IDs where flag_positive changes within the visit: {len(visits_with_changes)}")
note = "flag_positive never changes within a single session_id"
analysis_notes.append(note)
print_analysis_note(note)

# Find the number of visit ids per day for each entity
visit_dates = df_data_sorted_by_date.groupby("session_id")[["timestamp", "entity"]].first().reset_index()
visit_dates["observation_day"] = visit_dates["timestamp"].dt.floor('d')

# Plot the distribution of visits per day for each entity
for entity in unique_entitys:
    entity_visit_dates = visit_dates[visit_dates["entity"] == entity]
    visits_per_day_entity = entity_visit_dates.groupby("observation_day")["session_id"].nunique().reset_index()

    plt.figure(figsize=(10, 5))
    plt.bar(visits_per_day_entity["observation_day"], visits_per_day_entity["session_id"], color='orange')
    plt.xlabel("Day")
    plt.ylabel("Number of Visit IDs")
    plt.title(f"Distribution of Visit IDs per Day - Client {entity}")
    plt.tight_layout()
    plt.show()

# Also plot all entitys together for comparison
visits_per_day = visit_dates.groupby("observation_day")["session_id"].nunique().reset_index()
plt.figure(figsize=(10, 5))
plt.bar(visits_per_day["observation_day"], visits_per_day["session_id"], color='orange')
plt.xlabel("Day")
plt.ylabel("Number of Visit IDs")
plt.title(f"Distribution of Visit IDs per Day (All Clients)")
plt.tight_layout()
plt.show()

# Plot the number of visits per day for a single patient (zoomed in)
filt = (visits_per_day["observation_day"] >= min_date + pd.Timedelta(days=100)) & \
    (visits_per_day["observation_day"] <= min_date + pd.Timedelta(days=130))
visits_per_day_filt = visits_per_day.loc[filt, :]
plt.figure(figsize=(10, 5))
plt.bar(visits_per_day_filt["observation_day"], visits_per_day_filt["session_id"], color='orange')
plt.xlabel("Day")
plt.ylabel("Number of Visit IDs")
plt.title(f"Number of Visit IDs per Day (Days 100-130)")
plt.tight_layout()
plt.show()

# All visit ids where condition is present
visits_with_condition = df_raw_data[df_raw_data["flag_positive"] == True]["session_id"].unique()
num_session_ids = df_raw_data["session_id"].nunique()
pct_visits_with_condition = len(visits_with_condition) / num_session_ids
print(f"Number of visit IDs where condition is present at least once: {len(visits_with_condition)}")
print(f"Percentage of visit IDs with condition present: {pct_visits_with_condition:.2%}")
note = "after aggregating to session_id level, condition is present in about 10% of visits"
analysis_notes.append(note)
print_analysis_note(note)

# Plot procal versus condition present for the sample entity
unique_patients = sorted(df_raw_data["entity"].unique())
sample_entity = unique_patients[0]
filt = (df_raw_data["entity"] == sample_entity) & (df_raw_data["metric_type"] == "procal")
entity_data_filt = df_raw_data.loc[filt, :]
entity_data_filt = entity_data_filt.sort_values(by="timestamp", ascending=True)
plt.figure(figsize=(10, 5))
#  Filter data to 10 days
filt = (entity_data_filt["timestamp"] >= min_date + pd.Timedelta(days=200)) & \
    (entity_data_filt["timestamp"] <= min_date + pd.Timedelta(days=210))
entity_data_filt = entity_data_filt.loc[filt, :]
#  Separate data based on flag_positive (after filtering)
condition_true = entity_data_filt[entity_data_filt["flag_positive"] == True]
#  Plot all procal values with black line
plt.plot(entity_data_filt["timestamp"], entity_data_filt["metric_value"],
         linestyle='-', marker='o', color='black', alpha=0.6, label='procal value')
#  Highlight points where condition IS present with red X
if not condition_true.empty:
    plt.scatter(condition_true["timestamp"], condition_true["metric_value"],
                marker='x', color='red', s=80,
                label='condition present', zorder=5)
plt.xlabel("Timestamp")
plt.ylabel("Procal Value")
plt.title(f"Procal Value and Condition Present over Time for Client {sample_entity}")
plt.legend()
plt.tight_layout()
plt.show()
note = "procal values tend to be higher when condition is present" + \
    " but are not a direct proxy for dvar"
analysis_notes.append(note)
print_analysis_note(note)

print("\n\nSummary of analysis notes:")
[print(f"- {n}") for n in analysis_notes]