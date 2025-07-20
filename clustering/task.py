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
    
    # Extract program and year level
    df[['Program', 'Year Level']] = df['Program & Year Level'].str.extract(r'([A-Z]+)\\s*-\\s*(\\d+)')
    df['Year Level'] = df['Year Level'].astype(int)
    
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
        'One or both deceased': 'Deceased',
        'Prefer not to say': 'Unknown'
    }
    df['Parents Marital Status'] = df['What is your parents\' current relationship status?'].map(marital_map)
    
    birth_order_map = {
        'Oldest': 'Oldest',
        'Middle Child': 'Middle',
        'Youngest': 'Youngest',
        'Only Child': 'Only'
    }
    df['Birth Order'] = df['What is your birth order among your siblings?'].map(birth_order_map)
    
    df['Has External Responsibilities'] = df['Do you have major responsibilities outside of school (e.g., part-time job, caregiving)? If yes, please describe briefly. Leave blank if none.'].apply(lambda x: 0 if x == 'Unknown' else 1)
    
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
    
    df_final = df[final_columns].copy()
    df_final = df_final.drop(columns=['Student ID'])
    
    # Encode categorical features
    label_encoders = {}
    for col in ['Program', 'Personality', 'Birth Order']:
        le = LabelEncoder()
        df_final[col] = le.fit_transform(df_final[col])
        label_encoders[col] = le
    
    # One-hot encode marital status
    df_final = pd.get_dummies(df_final, columns=['Parents Marital Status'], prefix='Marital', dtype=int)
    
    # Scale numerical features
    numeric_cols = ['Year Level', 'Academic Performance Change', 'Workload Rating', 
                   'Hobby Count', 'Financial Status', 'Average']
    scaler = StandardScaler()
    df_final[numeric_cols] = scaler.fit_transform(df_final[numeric_cols])
    
    # Drop some columns (adjust as needed)
    df_final = df_final.drop(columns=['Program', 'Year Level', 'Gender'])
    
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
    
    # Convert to list of dictionaries
    students_list = processed_df.to_dict('records')
    
    # Optional: Round float values for cleaner output
    for student in students_list:
        for key, value in student.items():
            if isinstance(value, (np.float32, np.float64)):
                student[key] = round(float(value), 2)
    
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