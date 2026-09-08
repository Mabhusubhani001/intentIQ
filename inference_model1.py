import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime

class NewbieClassifier:
    def __init__(self, model_path):
        print(f"Loading Model 1 from {model_path}...")
        self.model = lgb.Booster(model_file=model_path)
        self.feature_names = [
            'zone_entropy', 'scroll_direction_flips', 'cursor_step_variability_normalized',
            'has_first_click', 'time_to_first_click', 'distinct_zones_visited',
            'interaction_rate', 'mousemove_rate', 'click_rate', 'scroll_rate'
        ]
        # Stateful aggregation for live telemetry streaming
        self.session_states = {}

    def extract_features_from_payload(self, data: dict):
        events = data.get('events', [])
        session_id = data.get('session_id', 'default')
        
        # Initialize session state if new
        if session_id not in self.session_states:
            self.session_states[session_id] = {
                'zone_counts': {},
                'last_sy': 0,
                'last_scroll_dir': 0,
                'scroll_direction_flips': 0,
                'step_distances': [],
                'has_first_click': 0,
                'time_to_first_click': 30000,
                'total_events': 0,
                'total_mousemoves': 0,
                'total_clicks': 0,
                'total_scroll_events': 0,
                'max_t': 0,
                'last_x': None,
                'last_y': None
            }
            
        state = self.session_states[session_id]
        state['total_events'] += len(events)
        
        for e in events:
            etype = e.get('e')
            t = e.get('t', 0)
            
            if t > state['max_t']:
                state['max_t'] = t
                
            # Zone tracking
            if 'z' in e and e['z']:
                z = e['z']
                state['zone_counts'][z] = state['zone_counts'].get(z, 0) + 1
                
            # Scroll flips
            if 'sy' in e:
                state['total_scroll_events'] += 1
                current_sy = e['sy']
                if current_sy != state['last_sy']:
                    current_dir = 1 if current_sy > state['last_sy'] else -1
                    if state['last_scroll_dir'] != 0 and current_dir != state['last_scroll_dir']:
                        state['scroll_direction_flips'] += 1
                    state['last_scroll_dir'] = current_dir
                state['last_sy'] = current_sy
                
            # Mouse moves
            if etype == 'mousemove':
                state['total_mousemoves'] += 1
                if 'x' in e and 'y' in e:
                    if state['last_x'] is not None and state['last_y'] is not None:
                        dx = e['x'] - state['last_x']
                        dy = e['y'] - state['last_y']
                        dist = np.sqrt(dx**2 + dy**2)
                        state['step_distances'].append(dist)
                    state['last_x'] = e['x']
                    state['last_y'] = e['y']
                    
            # Clicks
            if etype == 'click':
                state['total_clicks'] += 1
                if state['has_first_click'] == 0:
                    state['has_first_click'] = 1
                    state['time_to_first_click'] = t

        # 1. zone_entropy
        total_zones = sum(state['zone_counts'].values())
        zone_entropy = 0
        if total_zones > 0:
            for count in state['zone_counts'].values():
                p = count / total_zones
                zone_entropy -= p * np.log2(p)
                
        # 2. scroll_direction_flips
        scroll_flips = state['scroll_direction_flips']
        
        # 3. cursor_step_variability_normalized
        if len(state['step_distances']) > 1:
            mean_dist = np.mean(state['step_distances'])
            std_dist = np.std(state['step_distances'])
            cursor_var = std_dist / mean_dist if mean_dist > 0 else 0
        else:
            cursor_var = 0
            
        # 4. has_first_click
        has_first = state['has_first_click']
        
        # 5. time_to_first_click
        time_to_first = state['time_to_first_click']
        
        # 6. distinct_zones_visited
        distinct_zones = len(state['zone_counts'])
        
        # Rates
        duration_s = max(state['max_t'] / 1000.0, 1.0)
        
        # 7-10. Rates
        interaction_rate = state['total_events'] / duration_s
        mousemove_rate = state['total_mousemoves'] / duration_s
        click_rate = state['total_clicks'] / duration_s
        scroll_rate = state['total_scroll_events'] / duration_s
        
        return [
            zone_entropy, scroll_flips, cursor_var, has_first, time_to_first,
            distinct_zones, interaction_rate, mousemove_rate, click_rate, scroll_rate
        ]

    def predict(self, data: dict):
        """
        Runs the full inference pipeline.
        Returns the routing signal.
        """
        # 1. Extract
        features = self.extract_features_from_payload(data)
        
        # 2. Format for LightGBM (requires 2D array/DataFrame)
        df = pd.DataFrame([features], columns=self.feature_names)
        
        # 3. Predict via ML
        prob = float(self.model.predict(df)[0])
        is_newbie = prob > 0.5
        
        # 4. Generate Signal
        signal = "SWITCH_TO_BEGINNER" if is_newbie else "PRO_LAYOUT"
        
        return {
            "session_id": data.get("session_id"),
            "probability": float(prob),
            "signal": signal,
            "features": dict(zip(self.feature_names, features))
        }

# --- Quick Local Test ---
if __name__ == "__main__":
    import os
    
    # Initialize the worker
    base_path = r"c:\Users\munvar\OneDrive\Desktop\data"
    model_path = os.path.join(base_path, 'model1_newbie_classifier.txt')
    classifier = NewbieClassifier(model_path)
    
    # Simulate receiving a live JSON payload over WebSocket
    test_file = os.path.join(base_path, "newbie", "newbie_browsing_session_mts746eo_xb1x25j.json")
    with open(test_file, 'r', encoding='utf-8') as f:
        live_payload = json.load(f)
    
    print("\n[Worker] Received telemetry batch for session:", live_payload.get("session_id"))
    result = classifier.predict(live_payload)
    print(f"[Worker] Prediction Result: {json.dumps(result, indent=2)}")
    
    # Test a veteran
    test_file_2 = os.path.join(base_path, "veteran", "veteran_direct_session_mtsalev6_2crbmdv.json")
    with open(test_file_2, 'r', encoding='utf-8') as f:
        live_payload_2 = json.load(f)
        
    print("\n[Worker] Received telemetry batch for session:", live_payload_2.get("session_id"))
    result_2 = classifier.predict(live_payload_2)
    print(f"[Worker] Prediction Result: {json.dumps(result_2, indent=2)}")
