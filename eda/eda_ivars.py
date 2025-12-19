"""
EDA for independent variables in the sepsis dataset.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

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

session_ids = sorted(df_raw_data["session_id"].unique())
session_id = session_ids[10]
df_tmp = df_raw_data.loc[df_raw_data["session_id"] == session_id, :]
print(df_tmp["timestamp"].min())
print(df_tmp["timestamp"].max())

# Time range
df_raw_data["timestamp"] = pd.to_datetime(df_raw_data["timestamp"])
min_date = df_raw_data["timestamp"].min()
max_date = df_raw_data["timestamp"].max()
print(f"Data ranges from {min_date} to {max_date}")

# Client ids
unique_entitys = sorted(df_raw_data["entity"].unique())
print(f"Unique entitys: {unique_entitys}")

# Initialize some analysis notes for aggregation later
analysis_notes = []


# INDEPENDENT VARIABLE ANALYSIS
# ===============================================================================
# Different types of observation codes
observation_codes = sorted(df_raw_data["metric_type"].unique())
print("Counts of each observation type code:")
print(df_raw_data["metric_type"].value_counts())
print(f"\nTotal unique observation type codes: {len(observation_codes)}")
note = "there are 18 independent variable observation types (ivars) present"
analysis_notes.append(note)
print_analysis_note(note)

# Average time resolution of the ivar data (just rough estimate, not accounting for visits)
avg_time_diffs = []
for observation_code in observation_codes:
    entity_data = df_raw_data[df_raw_data["metric_type"] == observation_code]
    entity_data = df_raw_data.sort_values(by="timestamp", ascending=True)
    time_diffs = entity_data["timestamp"].diff().dropna()
    avg_time_diff = time_diffs.mean()
    avg_time_diffs.append(avg_time_diff)
overall_avg_time_diff = sum(avg_time_diffs, pd.Timedelta(0)) / len(avg_time_diffs)
note = "average time difference between ivar observations is roughly 3 seconds"
analysis_notes.append(note)
print_analysis_note(note)

# Reference ranges for each observation type code
reference_ranges = df_raw_data.groupby("metric_type")["threshold_lower"].unique()
print("Reference ranges for each observation type code:")
for code, ranges in reference_ranges.items():
    print(f"  {code}: {ranges}")
note = "reference ranges are provided for most ivars but can vary significantly"
analysis_notes.append(note)
print_analysis_note(note)

# Total number of ivars present by visit ids
num_ivars_by_visit = df_raw_data.groupby("session_id")["metric_type"].nunique()
num_ivars_by_visit.sort_values(ascending=False, inplace=True)
ivar_count_distribution = Counter(num_ivars_by_visit)
print("\nDistribution of number of unique observation types per visit ID:")
for ivar_count, visit_count in sorted(ivar_count_distribution.items()):
    print(f"  {ivar_count} unique observation types: {visit_count} visits") 
note = "most visits have between 5 and 10 unique observation types recorded" + \
    " but counts vary significantly"
analysis_notes.append(note)
print_analysis_note(note)

# Find a visit id where condition is / is not present at least once for the first entity id
entity_id = unique_entitys[0]
entity_data = df_raw_data[df_raw_data["entity"] == entity_id]
visits_with_condition = entity_data[entity_data["flag_positive"] == True]["session_id"].unique()
visits_without_condition = entity_data[entity_data["flag_positive"] == False]["session_id"].unique()

# Scale the data to each observation value range for better visualization
for observation_code in observation_codes:
    obs_data = entity_data[entity_data["metric_type"] == observation_code]
    min_val = obs_data["metric_value"].min()
    max_val = obs_data["metric_value"].max()
    range_val = max_val - min_val
    if range_val > 0:
        scaled_values = (obs_data["metric_value"] - min_val) / range_val
        entity_data.loc[obs_data.index, "metric_value"] = scaled_values

# Plot observation data for a single visit id with condition present
session_id_with = visits_with_condition[0]
visit_data_with = entity_data[entity_data["session_id"] == session_id_with]
visit_data_with = visit_data_with.sort_values(by="timestamp", ascending=True)
plt.figure(figsize=(12, 6))
for observation_code in observation_codes:
    condition_data = visit_data_with[visit_data_with["metric_type"] == observation_code]
    plt.plot(condition_data["timestamp"], condition_data["metric_value"],
                label=observation_code, linewidth=2)
plt.xlabel("Timestamp")
plt.ylabel("Numeric Observation Value")
plt.title(f"Conditions over Time for Client {entity_id} (Visit ID {session_id_with} with Condition Present)")
plt.legend()
plt.tight_layout()
plt.show()

# Plot observation data for a single visit id without condition present
session_id_without = visits_without_condition[0]
visit_data_without = entity_data[entity_data["session_id"] == session_id_without]
visit_data_without = visit_data_without.sort_values(by="timestamp", ascending=True)
plt.figure(figsize=(12, 6))
for observation_code in observation_codes:
    condition_data = visit_data_without[visit_data_without["metric_type"] == observation_code]
    plt.plot(condition_data["timestamp"], condition_data["metric_value"],
                label=observation_code, linewidth=2)
plt.xlabel("Timestamp")
plt.ylabel("Numeric Observation Value")
plt.title(f"Conditions over Time for Client {entity_id} (Visit ID {session_id_without} without Condition Present)")
plt.legend()
plt.tight_layout()
plt.show()

# Plot distributions of observation values for each observation type code in a single figure
plt.figure(figsize=(15, 10))
for i, observation_code in enumerate(observation_codes):
    plt.subplot(4, 5, i + 1)
    obs_data = df_raw_data[df_raw_data["metric_type"] == observation_code]
    plt.hist(obs_data["metric_value"].dropna(), bins=20, color='blue', alpha=0.7)
    plt.title(f"Obs Type: {observation_code}")
    plt.xlabel("Observation Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
plt.suptitle("Distributions of Observation Values by Type Code", y=1.02)
plt.show()

note = "ivar data looks relatively clean we can proceed with feature engineering based on these observations"
analysis_notes.append(note)
print_analysis_note(note)

print("\n\nSummary of analysis notes:")
[print(f"- {n}") for n in analysis_notes]
