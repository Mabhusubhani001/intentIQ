import os
import json
import pandas as pd
from features_model2 import extract_features_model2

def sliding_window_features(events, label, window_ms=5000, step_ms=1000):
    if not events:
        return []
    
    t_start = events[0].get('t', 0)
    t_end = events[-1].get('t', 0)
    
    dataset = []
    
    # Slide a window across the session
    current_time = t_start
    while current_time + window_ms <= t_end:
        # Get events in the current window
        window_events = [e for e in events if current_time <= e.get('t', 0) < current_time + window_ms]
        
        if len(window_events) > 0:
            features = extract_features_model2(window_events)
            features['label'] = label
            dataset.append(features)
        
        current_time += step_ms
        
    return dataset

def build_dataset_from_new_data():
    data_dir = 'new_data'
    all_data = []
    
    for filename in os.listdir(data_dir):
        if not filename.endswith('.json'):
            continue
            
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r') as f:
            session_data = json.load(f)
            
        events = session_data.get('events', [])
        if not events:
            continue
            
        # Determine label based on filename
        if 'erratic' in filename:
            label = 2
        elif 'hesitant' in filename:
            label = 1
        elif 'newbie' in filename or 'veteran' in filename:
            label = 0
        else:
            continue
            
        print(f"Processing {filename} (Label {label}) with {len(events)} events...")
        session_dataset = sliding_window_features(events, label)
        all_data.extend(session_dataset)
        
    df = pd.DataFrame(all_data)
    print(f"Total samples extracted: {len(df)}")
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/model2_real_data.csv', index=False)
    print("Saved to data/model2_real_data.csv")

if __name__ == '__main__':
    build_dataset_from_new_data()
