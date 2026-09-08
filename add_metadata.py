import os
import json
import random
from datetime import datetime, timedelta

def random_date(start_date, end_date):
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    if days_between_dates <= 0:
        return start_date
    random_number_of_days = random.randrange(days_between_dates)
    return start_date + timedelta(days=random_number_of_days)

current_date = datetime(2026, 9, 8)
base_path = r"c:\Users\munvar\OneDrive\Desktop\data"

categories = {
    "newbie": {
        "creation_start": current_date - timedelta(days=7),
        "creation_end": current_date,
        "sessions": lambda: random.randint(1, 3),
        "bets": lambda s: random.randint(0, 1)
    },
    "hesistant": {
        "creation_start": current_date - timedelta(days=365),
        "creation_end": current_date - timedelta(days=30),
        "sessions": lambda: random.randint(5, 20),
        "bets": lambda s: random.randint(0, 2)
    },
    "veteran": {
        "creation_start": current_date - timedelta(days=1000),
        "creation_end": current_date - timedelta(days=365),
        "sessions": lambda: random.randint(50, 300),
        "bets": lambda s: s + random.randint(10, 100)
    },
    "erratic": {
        "creation_start": current_date - timedelta(days=500),
        "creation_end": current_date - timedelta(days=30),
        "sessions": lambda: random.randint(10, 100),
        "bets": lambda s: int(s * random.uniform(0.5, 2.0))
    }
}

for cat, params in categories.items():
    folder_path = os.path.join(base_path, cat)
    if not os.path.exists(folder_path):
        continue
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            sessions = params["sessions"]()
            bets = params["bets"](sessions)
            creation_date = random_date(params["creation_start"], params["creation_end"])
            
            data["metadata"]["account_creation_date"] = creation_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            data["metadata"]["no_of_active_sessions"] = sessions
            data["metadata"]["no_bets_placed"] = bets
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

print("Finished updating JSON files.")
