import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def extract_features(file_path, current_date=datetime(2026, 9, 8)):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    events = data.get('events', [])
    
    # 2. Mouse Features (Full Session/Batch)
    early_events = events
    
    zone_counts = {}
    total_scroll = 0
    last_sy = 0
    mouse_moves = []
    
    for e in early_events:
        etype = e.get('e')
        
        # Zone tracking
        if 'z' in e and e['z']:
            z = e['z']
            zone_counts[z] = zone_counts.get(z, 0) + 1
            
        # Scroll tracking
        if 'sy' in e:
            total_scroll += abs(e['sy'] - last_sy)
            last_sy = e['sy']
            
        # Mouse movement for jitter
        if etype == 'mousemove' and 'x' in e and 'y' in e:
            mouse_moves.append((e['x'], e['y']))
            
    # Calculate Zone Entropy
    total_zone_events = sum(zone_counts.values())
    zone_entropy = 0
    if total_zone_events > 0:
        for count in zone_counts.values():
            p = count / total_zone_events
            zone_entropy -= p * np.log2(p)
            
    # Calculate Cursor Jitter (standard deviation of step distances)
    distances = []
    for i in range(1, len(mouse_moves)):
        dx = mouse_moves[i][0] - mouse_moves[i-1][0]
        dy = mouse_moves[i][1] - mouse_moves[i-1][1]
        distances.append(np.sqrt(dx**2 + dy**2))
    
    cursor_jitter = np.std(distances) if len(distances) > 1 else 0
    
    click_count_early = sum(1 for e in early_events if e.get('e') == 'click')
    
    # Target Label: 1 for newbie, 0 for others
    is_newbie = 1 if data.get('label', '').startswith('newbie') else 0
    
    return {
        'zone_entropy': zone_entropy,
        'cursor_jitter': cursor_jitter,
        'total_scroll_early': total_scroll,
        'click_count_early': click_count_early,
        'is_newbie': is_newbie,
        'session_id': data.get('session_id', 'unknown')
    }

print("Loading and extracting features from raw JSON...")
base_path = r"c:\Users\munvar\OneDrive\Desktop\feg-solver\backend\data"
categories = ['newbie', 'hesistant', 'veteran', 'erratic']

all_features = []

for cat in categories:
    folder_path = os.path.join(base_path, cat)
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                try:
                    feat = extract_features(file_path)
                    all_features.append(feat)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

df = pd.DataFrame(all_features)
print(f"Extracted features for {len(df)} sessions.")

# Prepare Data
X = df.drop(['is_newbie', 'session_id'], axis=1)
y = df['is_newbie']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 10,
    'max_depth': 4,
    'feature_fraction': 0.8,
    'min_data_in_leaf':2,
    'scale_pos_weight': 3.0,
    'verbose': -1
}

print("Starting training...")
gbm = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[train_data, test_data],
    callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(10)]
)

print("\nModel training completed!")

y_pred_prob = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

importance = pd.DataFrame({
    'feature': X.columns,
    'importance': gbm.feature_importance()
}).sort_values('importance', ascending=False)
print("\nFeature Importances:")
print(importance)

# Save to feg-solver/backend to overwrite the existing model
model_save_path = r"c:\Users\munvar\OneDrive\Desktop\feg-solver\backend\model1_newbie_classifier.txt"
gbm.save_model(model_save_path)
print(f"\nModel successfully saved to: {model_save_path}")
