import pandas as pd
from datetime import datetime

# List of your CSV files
files = ["data/data_1.csv", "data/data_2.csv"]

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

for file in files:
    # Load the dataset
    df = pd.read_csv(file)
    
    # Add a new column 'flower_id' with sequential values starting at 1
    df.insert(0, "flower_id", range(1, len(df) + 1))
    df["event_timestamp"] = current_time
    
    # Save back to the same file (overwrite)
    df.to_csv(file, index=False)

    print(f"Updated {file} with flower_id column.")
