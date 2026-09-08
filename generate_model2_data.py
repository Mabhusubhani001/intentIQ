import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_samples=3000):
    data = []
    samples_per_class = num_samples // 3

    for _ in range(samples_per_class):
        # Class 0: Normal Play (low dwell, low clicks, low jitter)
        data.append({
            'cta_dwell_time': np.random.uniform(0, 2000),
            'click_velocity': np.random.uniform(0.0, 0.5),
            'slip_jitter': np.random.uniform(0, 5),
            'zone_transitions_rolling': np.random.randint(0, 4),
            'label': 0
        })
        
        # Class 1: Final-Step Hesitation (high dwell > 2000, low click, low/med jitter, low transitions)
        data.append({
            'cta_dwell_time': np.random.uniform(2500, 10000),
            'click_velocity': np.random.uniform(0.0, 0.3),
            'slip_jitter': np.random.uniform(0, 10),
            'zone_transitions_rolling': np.random.randint(0, 3),
            'label': 1
        })
        
        # Class 2: At-Risk/Erratic (high click velocity > 1.5 OR high zone transitions > 4)
        data.append({
            'cta_dwell_time': np.random.uniform(0, 1500),
            'click_velocity': np.random.uniform(1.5, 6.0),
            'slip_jitter': np.random.uniform(5, 20),
            'zone_transitions_rolling': np.random.randint(5, 12),
            'label': 2
        })

    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/model2_synthetic_data.csv', index=False)
    print(f"Generated {len(df)} rows and saved to data/model2_synthetic_data.csv")

if __name__ == '__main__':
    generate_synthetic_data()
