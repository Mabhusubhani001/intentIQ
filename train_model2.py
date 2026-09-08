import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import os

def train():
    print("Loading synthetic data...")
    df = pd.read_csv('data/model2_real_data.csv')
    
    X = df[['cta_dwell_time', 'click_velocity', 'slip_jitter', 'zone_transitions_rolling']]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Multi-Class LightGBM Model...")
    clf = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        random_state=42,
        n_estimators=100
    )
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    # Using macro avg for multi-class
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    
    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importances:")
    print(importance)
    
    os.makedirs('models', exist_ok=True)
    model_path = 'models/model2_intent.txt'
    clf.booster_.save_model(model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == '__main__':
    train()
