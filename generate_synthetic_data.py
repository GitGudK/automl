"""
Generate synthetic data for AutoML project
Creates placeholder data with the same schema as the original dataset
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration - reduced for smaller file size (GitHub friendly)
N_CLIENTS = 3
N_VISITS = 500  # Reduced from 5000
N_OBSERVATIONS_PER_VISIT = (20, 80)  # Reduced from (50, 200)
CONDITION_POSITIVE_RATE = 0.10  # 10% positive class

# Observation types with their typical ranges (obfuscated names)
OBSERVATION_TYPES = {
    'pulseRate': {'min': 50, 'max': 180, 'ref_min': None, 'ref_max': None},
    'bpUpper': {'min': 80, 'max': 200, 'ref_min': None, 'ref_max': None},
    'bpLower': {'min': 40, 'max': 120, 'ref_min': None, 'ref_max': None},
    'bodyTemp': {'min': 95.0, 'max': 105.0, 'ref_min': None, 'ref_max': None},
    'tempF': {'min': 95.0, 'max': 105.0, 'ref_min': None, 'ref_max': None},
    'tempC': {'min': 35.0, 'max': 41.0, 'ref_min': None, 'ref_max': None},
    'pltCount': {'min': 50, 'max': 450, 'ref_min': 150, 'ref_max': 400},
    'leucocyteCount': {'min': 2.0, 'max': 30.0, 'ref_min': 4.0, 'ref_max': 11.0},
    'immaturePct': {'min': 0, 'max': 30, 'ref_min': 0, 'ref_max': 10},
    'immatureCount': {'min': 0, 'max': 5, 'ref_min': 0, 'ref_max': 1},
    'renalMarker': {'min': 0.3, 'max': 10.0, 'ref_min': 0.7, 'ref_max': 1.3},
    'liverEnzA': {'min': 5, 'max': 500, 'ref_min': 7, 'ref_max': 56},
    'liverEnzB': {'min': 5, 'max': 500, 'ref_min': 10, 'ref_max': 40},
    'lactateLevel': {'min': 0.2, 'max': 10.0, 'ref_min': 0.36, 'ref_max': 1.25},
    'cardiacMarker': {'min': 0.0, 'max': 50.0, 'ref_min': None, 'ref_max': 0.04},
    'natriureticPeptide': {'min': 0, 'max': 5000, 'ref_min': None, 'ref_max': 100},
    'inflammMarker': {'min': 0.0, 'max': 200.0, 'ref_min': None, 'ref_max': 3.0},
    'infectionMarker': {'min': 0.0, 'max': 100.0, 'ref_min': None, 'ref_max': 0.15},
}

def generate_visit_data(session_id, entity, flag_positive, base_date):
    """Generate observations for a single visit"""
    observations = []

    # Determine number of observations for this visit
    n_obs = np.random.randint(*N_OBSERVATIONS_PER_VISIT)

    # Select random observation types (weighted toward common vitals)
    obs_types = list(OBSERVATION_TYPES.keys())
    weights = [2.0 if obs in ['pulseRate', 'bodyTemp', 'bpUpper', 'bpLower'] else 1.0
               for obs in obs_types]
    selected_types = random.choices(obs_types, weights=weights, k=n_obs)

    # Generate observations spread over time
    for i, obs_type in enumerate(selected_types):
        obs_config = OBSERVATION_TYPES[obs_type]

        # Add some noise to values - higher values for flag_positive
        if flag_positive:
            # Shift abnormal values for positive cases
            if obs_type in ['leucocyteCount', 'immaturePct', 'immatureCount', 'renalMarker', 'liverEnzA', 'liverEnzB',
                           'lactateLevel', 'cardiacMarker', 'natriureticPeptide', 'inflammMarker', 'infectionMarker', 'pulseRate']:
                value = np.random.uniform(obs_config['min'] * 1.2, obs_config['max'])
            elif obs_type in ['pltCount']:
                value = np.random.uniform(obs_config['min'], obs_config['max'] * 0.8)
            else:
                value = np.random.uniform(obs_config['min'], obs_config['max'])
        else:
            # Normal distribution around healthy range
            value = np.random.uniform(obs_config['min'], obs_config['max'])

        # Round based on type
        if obs_type in ['pulseRate', 'pltCount', 'immatureCount', 'immaturePct', 'natriureticPeptide', 'liverEnzA', 'liverEnzB']:
            value = round(value)
        else:
            value = round(value, 2)

        # Generate timestamp (spread over ~7 days)
        hours_offset = np.random.uniform(0, 168)  # 7 days
        obs_date = base_date + timedelta(hours=hours_offset)

        observations.append({
            'entity': entity,
            'session_id': session_id,
            'metric_type': obs_type,
            'metric_value': value,
            'threshold_lower': obs_config['ref_min'],
            'threshold_upper': obs_config['ref_max'],
            'flag_positive': flag_positive,
            'timestamp': obs_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        })

    return observations

def main():
    print("Generating synthetic medical data...")
    print(f"  Clients: {N_CLIENTS}")
    print(f"  Visits: {N_VISITS}")
    print(f"  Condition positive rate: {CONDITION_POSITIVE_RATE:.0%}")

    all_observations = []
    entitys = ['entity_A', 'entity_B', 'entity_C'][:N_CLIENTS]

    # Generate visits
    for session_idx in range(N_VISITS):
        session_id = 1000000 + session_idx
        entity = random.choice(entitys)
        flag_positive = random.random() < CONDITION_POSITIVE_RATE
        base_date = datetime(2022, 1, 1) + timedelta(days=random.randint(0, 365))

        visit_obs = generate_visit_data(
            session_id=session_id,
            entity=entity,
            flag_positive=flag_positive,
            base_date=base_date
        )
        all_observations.extend(visit_obs)

        if (session_idx + 1) % 1000 == 0:
            print(f"  Generated {session_idx + 1}/{N_VISITS} visits...")

    # Create DataFrame
    df = pd.DataFrame(all_observations)

    # Shuffle rows
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to CSV
    output_path = 'data/data_science_project_data.csv'
    df.to_csv(output_path, index=False)

    print(f"\n✓ Synthetic data saved to {output_path}")
    print(f"  Total observations: {len(df):,}")
    print(f"  Unique visits: {df['session_id'].nunique():,}")
    print(f"  Positive cases: {df.groupby('session_id')['flag_positive'].first().sum():,} ({df.groupby('session_id')['flag_positive'].first().mean():.1%})")
    print(f"  Observation types: {df['metric_type'].nunique()}")

    # Show sample
    print("\nSample data:")
    print(df.head())

    print("\nData statistics:")
    print(df.describe())

if __name__ == "__main__":
    main()
