import json
import os

cells_code = [
'''import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib''',

'''def extract_features(file_path, current_date=datetime(2026, 9, 8)):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    events = data.get('events', [])
    
    # 1. Historical Features
    created_str = metadata.get('account_creation_date', current_date.strftime('%Y-%m-%dT%H:%M:%SZ'))
    created_date = datetime.strptime(created_str, '%Y-%m-%dT%H:%M:%SZ')
    account_age_days = max(1, (current_date - created_date).days)
    
    sessions = metadata.get('no_of_active_sessions', 1)
    bets = metadata.get('no_bets_placed', 0)
    
    bets_per_session = bets / sessions
    session_frequency = sessions / account_age_days
    
    # 2. Mouse Features (First 8000 ms)
    early_events = [e for e in events if e.get('t', 0) <= 8000]
    
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
        'account_age_days': account_age_days,
        'bets_per_session': bets_per_session,
        'session_frequency': session_frequency,
        'zone_entropy': zone_entropy,
        'cursor_jitter': cursor_jitter,
        'total_scroll_early': total_scroll,
        'click_count_early': click_count_early,
        'is_newbie': is_newbie,
        'session_id': data.get('session_id', 'unknown')
    }''',

'''base_path = "c:/Users/munvar/OneDrive/Desktop/data"
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
df.head()''',

'''X = df.drop(['is_newbie', 'session_id'], axis=1)
y = df['is_newbie']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")''',

'''train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.8,
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

print("\\nModel training completed!")''',

'''y_pred_prob = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

importance = pd.DataFrame({
    'feature': X.columns,
    'importance': gbm.feature_importance()
}).sort_values('importance', ascending=False)
print("\\nFeature Importances:")
print(importance)

model_path = os.path.join(base_path, 'model1_newbie_classifier.txt')
gbm.save_model(model_path)
print(f"\\nModel successfully saved to: {model_path}")'''
]

nb = {
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

for c in cells_code:
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\\n" for line in c.split('\\n')]
    }
    # remove trailing newline from last line
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\\n")
    nb["cells"].append(cell)

with open(r"c:\Users\munvar\OneDrive\Desktop\data\model1_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Notebook generated successfully.")
