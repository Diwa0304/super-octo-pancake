import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Define paths
DATA_DIR = Path("data")
PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("artifacts")

# Ensure directories exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. Custom Imputation Logic ---

class ConditionalMeanImputer:
    """
    Imputes missing feature values based on the mean of the last 10 available 
    samples of the same species. If fewer than 10 are available, uses all 
    available samples for that species up to that point.
    """
    def __init__(self, species_col='species'):
        self.species_col = species_col
        self.imputation_means = {} # Stores the calculated means for each feature/species pair
        self.target_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    def fit_transform(self, df):
        """Processes the DataFrame row by row to apply and save imputation."""
        df_imputed = df.copy()

        for species in df[self.species_col].unique():
            species_data = df_imputed[df_imputed[self.species_col] == species]
            
            # This dictionary will store the running history of non-missing values
            history = {feat: [] for feat in self.target_features}
            
            # This will store the final mean for this species/feature, used for inference
            self.imputation_means[species] = {}

            for index in species_data.index:
                
                for feature in self.target_features:
                    value = df_imputed.loc[index, feature]
                    
                    if pd.isna(value):
                        # Calculate mean based on last 10 (or less) non-missing samples
                        last_n = history[feature][-10:]
                        
                        if last_n:
                            imputation_value = np.mean(last_n)
                        else:
                            # Edge case: If no history for this species/feature, use the global mean of available data for that species
                            imputation_value = df[df[self.species_col] == species][feature].mean()
                            if np.isnan(imputation_value):
                                imputation_value = df[feature].mean() # Fallback to global feature mean if species has no data

                        df_imputed.loc[index, feature] = imputation_value
                    
                    else:
                        # Update history with the current non-missing value
                        history[feature].append(value)
            
            # Store the final imputation mean (mean of all data for this species) for inference use
            for feature in self.target_features:
                self.imputation_means[species][feature] = df_imputed[df_imputed[self.species_col] == species][feature].mean()

        return df_imputed

# --- 2. Main Preprocessing Function ---

def preprocess_data_flexible(file_paths, output_name,label_encoder_obj, save_encoders=True):
    """Loads data, applies label encoding, and performs conditional imputation."""
    
    # 1. Load and Concatenate Data
    df_list = [pd.read_csv(p) for p in file_paths]
    df = pd.concat(df_list, ignore_index=True)
    
    print(f"Loaded and merged {len(file_paths)} datasets into a DataFrame of size {len(df)}.")

    # The 'species' column is guaranteed to be present and non-missing for labeling
    df['target'] = label_encoder_obj.transform(df['species'])    
    
    # 3. Custom Imputation (Feature Variables)
    imputer = ConditionalMeanImputer(species_col='species')
    df_processed = imputer.fit_transform(df)

    # 4. Final Cleanup and Output Selection
    # Select only the features and the new target column
    output_df = df_processed[[
        'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'
    ]]
    
    # 5. Save Artifacts and Data
    if save_encoders:
        # Save the label encoder and imputer means for production
        joblib.dump(imputer.imputation_means, ARTIFACTS_DIR / f'imputer_means_{output_name}.joblib')
        print(f"Saved LabelEncoder and ImputerMeans to {ARTIFACTS_DIR}")

    # Save the processed data file
    output_path = PROCESSED_DIR / f'{output_name}_processed.csv'
    output_df.to_csv(output_path, index=False)
    print(f"Successfully saved processed data to {output_path}")

# --- 3. Execution (Matching Your Iterations) ---

if __name__ == '__main__':
    v1_raw_path = DATA_DIR / 'iris_v1.csv'
    v2_raw_path = DATA_DIR / 'iris_v2.csv'
    
    # 1. FIT MASTER LABEL ENCODER ON ALL DATA FIRST (for universal mapping)
    print("\n--- 1. Fitting Master Label Encoder ---")
    
    # Load all raw data just to fit the LE, ensuring consistent mapping across all versions
    df_v1 = pd.read_csv(v1_raw_path)
    df_v2 = pd.read_csv(v2_raw_path)
    df_combined = pd.concat([df_v1, df_v2], ignore_index=True)

    master_le = LabelEncoder()
    master_le.fit(df_combined['species'])
    
    # Save the master Label Encoder
    joblib.dump(master_le, ARTIFACTS_DIR / 'label_encoder_master.joblib')
    print(f"Saved MASTER LabelEncoder (fitted on V1+V2) to {ARTIFACTS_DIR / 'label_encoder_master.joblib'}")


    # 2. PROCESS V1 SEPARATELY (using the master encoder)
    print("\n--- 2. Processing V1 Data ---")
    # Assuming the function is named 'preprocess_version' or the flexible 'preprocess_data_flexible' 
    # as defined in the previous response, which accepts the master_le object.
    preprocess_data_flexible(
        file_paths=[v1_raw_path],
        output_name='iris_v1',
        label_encoder_obj=master_le,  
        save_encoders=True 
    )

    # 3. PROCESS V2 SEPARATELY (using the master encoder)
    print("\n--- 3. Processing V2 Data ---")
    preprocess_data_flexible(
        file_paths=[v2_raw_path],
        output_name='iris_v2',
        label_encoder_obj=master_le,
        save_encoders=True 
    )
    
    print("\nPreprocessing complete. Both V1 and V2 processed data files and specific imputer means are saved.")