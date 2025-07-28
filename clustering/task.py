import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_initial_clean(filepath):
    """Load raw data and perform initial cleaning."""
    try:
        raw_data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}. Please check the path.")
    
    # Drop columns with all NaN values
    df_cleaned = raw_data.dropna(axis=1, how='all')
    
    # Split 'Program & Year Level' into separate columns
    df_cleaned[['program', 'year']] = df_cleaned['Program & Year Level'].str.split('-', expand=True)
    df_cleaned['year'] = df_cleaned['year'].astype(int)
    
    # Sort and drop unnecessary columns
    df_sorted = df_cleaned.sort_values(by=['program', 'year'])
    df_sorted = df_sorted.drop(columns=df_sorted.columns[0:2])
    
    return df_sorted


def process_grades(df):
    """Discretize grades and calculate averages."""
    # Select grade columns (adjust indices as needed)
    grade_cols_1 = df.columns[3:58]
    grade_cols_2 = df.columns[68:74]
    grade_cols = grade_cols_1.union(grade_cols_2)
    
    df_grade_cols = df[grade_cols]
    
    # Map grade ranges to numerical values
    grade_mapping = {
        '1.0 - 1.2': 0,
        '1.21 - 1.4': 1,
        '1.41 - 2.0': 2,
        '2.1 - 2.5': 3,
        '2.51 - 3.0': 4
    }
    
    for col in df_grade_cols:
        df_grade_cols[col] = df_grade_cols[col].map(grade_mapping)
        df_grade_cols[col] = df_grade_cols[col].fillna(-1)
    
    df_grade_cols['Average'] = df_grade_cols.apply(lambda row: row[row != -1].mean(), axis=1)
    
    # Merge with main dataframe
    df_result = df.drop(columns=grade_cols).reset_index(drop=True)
    df_result['Average'] = df_grade_cols['Average']
    
    return df_result


def clean_and_transform(df):
    """Clean data and transform categorical features."""
    # Clean column names and values
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
    df.replace('', np.nan, inplace=True)
    df.fillna('Unknown', inplace=True)
    
    # Extract program and year level with error handling
    if 'Program & Year Level' in df.columns:
        extraction = df['Program & Year Level'].str.extract(r'([A-Z]+)\\s*-\\s*(\\d+)')
        df['Program'] = extraction[0].fillna('Unknown')
        df['Year Level'] = pd.to_numeric(extraction[1], errors='coerce').fillna(1).astype(int)
    else:
        df['Program'] = 'Unknown'
        df['Year Level'] = 1
    
    # Map categorical features to numerical values
    performance_map = {
        'Significantly Improved': 2,
        'Slightly Improved': 1,
        'Stayed the Same': 0,
        'Slightly Declined': -1,
        'Significantly Declined': -2
    }
    df['Academic Performance Change'] = df['How do you feel your academic performance has changed this year?'].map(performance_map)
    
    workload_map = {
        'Very Easy': 1,
        'Easy': 2,
        'Moderate': 3,
        'Challenging': 4,
        'Very Challenging': 5
    }
    df['Workload Rating'] = df['How would you rate your current academic workload?'].map(workload_map)
    
    df['Gender'] = df['Gender'].apply(lambda x: 1 if str(x).lower() == 'female' else (0 if str(x).lower() == 'male' else 2))
    
    # Learning styles (one-hot encoding)
    learning_styles = ['Visual', 'Auditory', 'Reading/Writing', 'Kinesthetic']
    for style in learning_styles:
        df[f'Learning_{style}'] = df['How do you prefer to learn? (Select all that apply)”'].str.contains(style).astype(int)
    
    help_map = {
        'Always asks for help immediately': 0,
        'Asks after trying for a while': 1,
        'Waits too long to ask': 2,
        'Prefers to solve problems alone': 3
    }
    df['Help Seeking'] = df['How do you usually handle academic challenges?'].map(help_map)
    
    # Extract personality type
    df['Personality'] = df['Which personality type best describes you?'].str.extract(r'(Introvert|Ambivert|Extrovert)')[0]
    df['Personality'] = df['Personality'].fillna('Unknown')
    
    # Count hobbies
    df['Hobby Count'] = df['What activities or hobbies do you enjoy outside of class? (Select/Write at least 3)'].str.split(',').apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    financial_map = {
        'Very Comfortable': 4,
        'Comfortable': 3,
        'Somewhat Challenging': 2,
        'Struggling': 1,
        'Severely Struggling': 0
    }
    df['Financial Status'] = df['How would you describe your current financial situation as a student?'].map(financial_map)
    
    marital_map = {
        'Together': 'Together',
        'Separated': 'Separated',
        'One or Both Parents Deceased': 'Deceased',
        'Prefer not to say': 'Unknown'
    }
    # Find marital status column dynamically
    marital_col = None
    for col in df.columns:
        if 'relationship status' in col.lower():
            marital_col = col
            break
    
    if marital_col and marital_col in df.columns:
        df['Parents Marital Status'] = df[marital_col].map(marital_map)
        # Fill any unmapped values with 'Unknown'
        df['Parents Marital Status'] = df['Parents Marital Status'].fillna('Unknown')
    else:
        df['Parents Marital Status'] = 'Unknown'
    
    birth_order_map = {
        'Oldest': 'Oldest',
        'Middle Child': 'Middle',
        'Youngest': 'Youngest',
        'Only Child': 'Only'
    }
    df['Birth Order'] = df['What is your birth order among your siblings?'].map(birth_order_map)
    
    # External responsibilities with safe handling
    resp_col = 'Do you have major responsibilities outside of school (e.g., part-time job, caregiving)? If yes, please describe briefly. Leave blank if none.'
    if resp_col in df.columns:
        df['Has External Responsibilities'] = df[resp_col].apply(
            lambda x: 0 if str(x).lower() in ['unknown', 'nan', ''] else 1
        )
    else:
        df['Has External Responsibilities'] = 0
    
    return df


def select_final_features(df):
    """Select and process final features for modeling."""
    final_columns = [
        'Student ID', 'Program', 'Year Level', 'Gender',
        'Academic Performance Change', 'Workload Rating',
        'Learning_Visual', 'Learning_Auditory', 'Learning_Reading/Writing', 'Learning_Kinesthetic',
        'Help Seeking', 'Personality', 'Hobby Count',
        'Financial Status', 'Parents Marital Status', 'Birth Order',
        'Has External Responsibilities', 'Average'
    ]
    
    # Only include columns that actually exist
    existing_columns = [col for col in final_columns if col in df.columns]
    df_final = df[existing_columns].copy()
    
    if 'Student ID' in df_final.columns:
        df_final = df_final.drop(columns=['Student ID'])
    
    # Handle NaN values before any processing
    print(f"DEBUG: Checking for NaN values before processing...")
    nan_counts = df_final.isnull().sum()
    for col, count in nan_counts[nan_counts > 0].items():
        print(f"DEBUG: Column '{col}' has {count} NaN values")
    
    # Fill NaN values with appropriate defaults
    # Numeric columns that can have NaN
    numeric_fill_cols = ['Academic Performance Change', 'Workload Rating', 'Help Seeking', 
                        'Hobby Count', 'Financial Status', 'Average']
    for col in numeric_fill_cols:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna(0.0)
    
    # Binary columns (learning styles, responsibilities)
    binary_cols = ['Learning_Visual', 'Learning_Auditory', 'Learning_Reading/Writing', 
                   'Learning_Kinesthetic', 'Has External Responsibilities']
    for col in binary_cols:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna(0)
    
    # Categorical columns
    categorical_cols = ['Program', 'Personality', 'Parents Marital Status', 'Birth Order']
    for col in categorical_cols:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna('Unknown')
    
    # Fill remaining NaN values
    df_final = df_final.fillna(0)
    
    # Encode categorical features
    label_encoders = {}
    for col in ['Program', 'Personality', 'Birth Order']:
        if col in df_final.columns:
            le = LabelEncoder()
            # Ensure no NaN values before encoding
            df_final[col] = df_final[col].astype(str).fillna('Unknown')
            df_final[col] = le.fit_transform(df_final[col])
            label_encoders[col] = le
    
    # One-hot encode marital status - ensure specific columns for cluster engine
    if 'Parents Marital Status' in df_final.columns:
        df_final['Parents Marital Status'] = df_final['Parents Marital Status'].astype(str).fillna('Unknown')
        
        # Create the specific marital status columns expected by cluster engine
        df_final['Marital_Separated'] = (df_final['Parents Marital Status'] == 'Separated').astype(int)
        df_final['Marital_Together'] = (df_final['Parents Marital Status'] == 'Together').astype(int)
        
        # Drop the original column
        df_final = df_final.drop(columns=['Parents Marital Status'])
    
    # Scale numerical features - handle NaN values first
    numeric_cols = ['Year Level', 'Academic Performance Change', 'Workload Rating', 
                   'Hobby Count', 'Financial Status', 'Average']
    
    # Ensure numeric columns exist and have no NaN
    existing_numeric_cols = [col for col in numeric_cols if col in df_final.columns]
    if existing_numeric_cols:
        # Fill any remaining NaN in numeric columns
        df_final[existing_numeric_cols] = df_final[existing_numeric_cols].fillna(0.0)
        
        # Ensure all values are finite
        for col in existing_numeric_cols:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0.0)
            # Replace infinite values
            df_final[col] = df_final[col].replace([np.inf, -np.inf], 0.0)
        
        scaler = StandardScaler()
        try:
            df_final[existing_numeric_cols] = scaler.fit_transform(df_final[existing_numeric_cols])
        except Exception as e:
            print(f"DEBUG: Scaling error: {e}")
            # If scaling fails, just normalize manually
            for col in existing_numeric_cols:
                mean_val = df_final[col].mean()
                std_val = df_final[col].std()
                if std_val > 0:
                    df_final[col] = (df_final[col] - mean_val) / std_val
                else:
                    df_final[col] = 0.0
            scaler = None
    else:
        scaler = StandardScaler()  # Empty scaler
    
    # Drop some columns (adjust as needed)
    cols_to_drop = ['Program', 'Year Level', 'Gender']
    existing_cols_to_drop = [col for col in cols_to_drop if col in df_final.columns]
    if existing_cols_to_drop:
        df_final = df_final.drop(columns=existing_cols_to_drop)
    
    # Final check for NaN values
    print(f"DEBUG: Final NaN check...")
    final_nan_counts = df_final.isnull().sum()
    if final_nan_counts.sum() > 0:
        print(f"DEBUG: Still have NaN values: {final_nan_counts[final_nan_counts > 0].to_dict()}")
        df_final = df_final.fillna(0.0)
    
    # Ensure all values are finite
    df_final = df_final.replace([np.inf, -np.inf], 0.0)
    
    print(f"DEBUG: Final dataframe shape: {df_final.shape}")
    print(f"DEBUG: Final columns: {list(df_final.columns)}")
    
    return df_final, label_encoders, scaler

def convert_to_api_format(processed_df):
    """
    Convert processed DataFrame to list of student dictionaries
    Args:
        processed_df: DataFrame from select_final_features()
    Returns:
        List of student records in API-friendly format
    """
    # Reset index to get sequential IDs (optional)
    processed_df = processed_df.reset_index(drop=True)
    
    # Replace any remaining NaN values with appropriate defaults
    processed_df = processed_df.fillna(0.0)
    
    # Convert to list of dictionaries
    students_list = processed_df.to_dict('records')
    
    # Clean and validate values for each student
    for student in students_list:
        for key, value in student.items():
            if pd.isna(value) or (isinstance(value, float) and value != value):
                # Handle NaN values based on field type
                if key.startswith('Learning_') or key.startswith('Marital_') or key == 'Has External Responsibilities':
                    student[key] = 0
                else:
                    student[key] = 0.0
            elif isinstance(value, (np.float32, np.float64)):
                # Round float values for cleaner output
                student[key] = round(float(value), 2)
            elif isinstance(value, (np.int32, np.int64)):
                # Convert numpy integers to Python int
                student[key] = int(value)
    
    return students_list

def preprocess_pipeline(filepath, return_format='dataframe'):
    """Complete preprocessing pipeline with format options."""
    df = load_and_initial_clean(filepath)
    df = process_grades(df)
    df = clean_and_transform(df)
    df_final, label_encoders, scaler = select_final_features(df)
    
    if return_format == 'api':
        students_list = convert_to_api_format(df_final)
        return students_list, label_encoders, scaler
    else:
        return df_final, label_encoders, scaler


if __name__ == "__main__":
    # Example usage
    processed_data, label_encoders, scaler = preprocess_pipeline('raw_data.csv')
    processed_data.to_csv('processed_student_data.csv', index=False)
    print("Preprocessing complete. Data saved to 'processed_student_data.csv'")

