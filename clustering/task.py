import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import uuid


def load_and_initial_clean(filepath):
    """Load raw data and perform initial cleaning."""
    try:
        raw_data = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}. Please check the path.")

    # Drop columns with all NaN values
    df_cleaned = raw_data.dropna(axis=1, how='all')

    # Store original 'Program & Year Level' for later use in Student model
    df_cleaned['Original Program & Year'] = df_cleaned.get('Program & Year Level', '').astype(str)

    if 'Program & Year Level' in df_cleaned.columns:
        extracted_data = df_cleaned['Program & Year Level'].astype(str).str.extract(r'([A-Za-z]+)\s*[-]?\s*(\d+)?')
        df_cleaned['program_str'] = extracted_data[0].fillna('Unknown_Program').astype(str)
        df_cleaned['year_int'] = pd.to_numeric(extracted_data[1], errors='coerce').fillna(0).astype(int)
    else:
        df_cleaned['program_str'] = 'Unknown_Program'
        df_cleaned['year_int'] = 0

    print(f"DEBUG (task.py - load_and_initial_clean): Sample program_str: {df_cleaned['program_str'].head().tolist()}")
    print(f"DEBUG (task.py - load_and_initial_clean): Sample year_int: {df_cleaned['year_int'].head().tolist()}")

    return df_cleaned

def clean_rename_and_select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the DataFrame, renames columns from questions to titles,
    and selects only the relevant columns for processing.

    Args:
        df: The raw pandas DataFrame.

    Returns:
        A cleaned DataFrame with specific columns selected and renamed.
    """
    # Drop columns that are entirely NaN
    df_cleaned = df.dropna(axis=1, how='all')

    # Split 'Program & Year Level' into 'Program' and 'Year Level'
    df_cleaned[['Program', 'Year Level']] = df_cleaned['Program & Year Level'].str.split('-', expand=True)
    df_cleaned['Year Level'] = pd.to_numeric(df_cleaned['Year Level'], errors='coerce').fillna(0).astype(int)

    # Define the mapping from original question-like columns to new titles
    column_rename_map = {
        'How do you feel your academic performance has changed this year?': 'Academic Performance Change',
        'How would you rate your current academic workload?': 'Workload Rating',
        'How do you usually handle academic challenges?': 'Help Seeking',
        'Which personality type best describes you?': 'Personality',
        'What activities or hobbies do you enjoy outside of class? (Select/Write at least 3)': 'Hobby Count',
        'How would you describe your current financial situation as a student?': 'Financial Status',
        'What is your parents’ current relationship status?': 'Parents Marital Status',
        'What is your birth order among your siblings?': 'Birth Order',
        'Do you have major responsibilities outside of school (e.g., part-time job, caregiving)? If yes, please describe briefly. Leave blank if none.': 'Has External Responsibilities'
    }
    df_cleaned.rename(columns=column_rename_map, inplace=True)

    learning_styles = ['Visual', 'Auditory', 'Reading/Writing', 'Kinesthetic']
    for style in learning_styles:
        df_cleaned[f'Learning_{style}'] = df_cleaned['How do you prefer to learn? (Select all that apply)'].str.contains(style).astype(int)

    df_cleaned.drop(columns=['How do you prefer to learn? (Select all that apply)'], inplace=True)

    # List of columns to keep for the final model
    columns_to_keep = [
        'Student ID', 'Name', 'Gender', 'program_str', 'year_int', 'Academic Performance Change',
        'Workload Rating', 'Learning_Visual', 'Learning_Auditory',
        'Learning_Reading/Writing', 'Learning_Kinesthetic', 'Help Seeking',
        'Personality', 'Hobby Count', 'Financial Status',
        'Parents Marital Status', 'Birth Order', 'Has External Responsibilities', 'Average'
    ]
    
    all_necessary_columns = columns_to_keep
    
    # Filter for only the columns we need
    df_selected = df_cleaned[[col for col in all_necessary_columns if col in df_cleaned.columns]]

    return df_selected

def apply_custom_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies custom mappings to categorical and ordinal features based on instructions.

    Args:
        df: The DataFrame with columns to be mapped.

    Returns:
        The DataFrame with features numerically mapped.
    """
    # Define mappings
    performance_map = {'Significantly Declined': -2, 'Slightly Declined': -1, 'Stayed the Same': 0, 'Slightly Improved': 1, 'Significantly Improved': 2}
    workload_map = {'Very Challenging': 0, 'Challenging': 1, 'Moderate': 2, 'Easy': 3, 'Very Easy': 4}
    help_seeking_map = {'Prefers to solve problems alone': 0, 'Waits too long to ask': 1, 'Asks after trying for a while': 2, 'Always asks for help immediately': 3}
    personality_map = {'Introvert': 0, 'Ambivert': 1, 'Extrovert': 2}
    financial_map = {'Severely Struggling': 0, 'Struggling': 1, 'Somewhat Challenging': 2, 'Comfortable': 3, 'Very Comfortable': 4}
    birth_order_map = {'Only Child': 0, 'Youngest': 1, 'Middle Child': 2, 'Oldest': 3}
    # marital_map = {'Prefer not to say': 0, 'One or both deceased': 1, 'Separated': 2, 'Together': 3}

    # Apply mappings
    df['Academic Performance Change'] = df['Academic Performance Change'].map(performance_map)
    df['Workload Rating'] = df['Workload Rating'].map(workload_map)
    df['Help Seeking'] = df['Help Seeking'].map(help_seeking_map)
    df['Personality'] = df['Personality'].map(personality_map)
    df['Financial Status'] = df['Financial Status'].map(financial_map)
    df['Birth Order'] = df['Birth Order'].map(birth_order_map)
    df['Has External Responsibilities'] = df['Has External Responsibilities'].apply(lambda x: 0 if x == 'Unknown' else 1)
    df['Hobby Count'] = df['Hobby Count'].str.split(',').apply(lambda x: len(x) if isinstance(x, list) else 0)

    # One-hot encode marital status and drop original column
    df['Marital_Together'] = (df['Parents Marital Status'] == 'Together').astype(int)
    df['Marital_Separated'] = (df['Parents Marital Status'] == 'Separated').astype(int)
    df.drop(columns=['Parents Marital Status'], inplace=True)

    return df

def scale_features(df: pd.DataFrame):
    """
    Scales all numerical columns using StandardScaler.

    Args:
        df: The DataFrame with numerical columns.

    Returns:
        The DataFrame with scaled numerical columns.
    """
    # Ensure all columns are numeric before scaling
    target_cols = ['Academic Performance Change', 'Workload Rating', 'Help Seeking', 'Financial Status', 'Birth Order', 'Personality', 'Hobby Count', 'Average']
    numeric_cols = df[target_cols].select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna(0, inplace=True) # Fill any NaNs that might have been introduced

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df_scaled, scaler

def convert_to_api_format(processed_df):
    """
    Convert processed DataFrame to list of student dictionaries,
    including original identifiers and correctly formatted program_and_grade.
    Args:
        processed_df: DataFrame from select_final_features()
    Returns:
        List of student records in API-friendly format
    """
    # Create 'Program and Year' for the Student model using original string/int values
    if 'program_str' in processed_df.columns and 'year_int' in processed_df.columns:
        # Ensure year_int is treated as an integer string for concatenation
        processed_df['Program and Year'] = processed_df['program_str'].astype(str) + processed_df['year_int'].astype(int).astype(str)
    else:
        processed_df['Program and Year'] = 'Unknown0'

    print(f"DEBUG (task.py - convert_to_api_format): Sample 'Program and Year' values before to_dict: {processed_df['Program and Year'].head().tolist()}")

    # The api_columns should include the raw 'program_str' and 'year_int'
    # if they are needed for constructing 'Program and Year' later,
    # but the final output should only have 'Program and Year'
    api_columns = [
        'Student ID', 'Name', 'Program and Year', # This is the final string for the Student model
        'Academic Performance Change', 'Workload Rating',
        'Learning_Visual', 'Learning_Auditory', 'Learning_Reading/Writing', 'Learning_Kinesthetic',
        'Help Seeking', 'Personality', 'Hobby Count', # 'Personality' is the encoded numerical value
        'Financial Status', 'Birth Order', 'Gender', # 'Birth Order' is the encoded numerical value
        'Has External Responsibilities', 'Average',
        'Marital_Separated', 'Marital_Together'
    ]

    existing_api_columns = [col for col in api_columns if col in processed_df.columns]
    df_api_format = processed_df[existing_api_columns].copy()

    for col in df_api_format.columns:
        if pd.api.types.is_numeric_dtype(df_api_format[col]):
            df_api_format[col] = df_api_format[col].fillna(0.0)
        else:
            df_api_format[col] = df_api_format[col].fillna('Unknown')

    students_list = df_api_format.to_dict('records')

    for student in students_list:
        for key, value in student.items():
            if pd.isna(value) or (isinstance(value, float) and value != value):
                if key.startswith('Learning_') or key.startswith('Marital_') or key == 'Has External Responsibilities' or key == 'Gender':
                    student[key] = 0
                else:
                    student[key] = 0.0
            elif isinstance(value, (np.float32, np.float64)):
                student[key] = round(float(value), 2)
            elif isinstance(value, (np.int32, np.int64)):
                student[key] = int(value)
            elif isinstance(value, str):
                student[key] = value.strip()

    print(f"DEBUG (task.py - convert_to_api_format): Sample 'Program and Year' values in final list: {[s.get('Program and Year') for s in students_list[:5]]}")

    return students_list

def save_results(df: pd.DataFrame, df_filepath: str):
    """
    Saves the processed DataFrame and the label encoders.
    """
    df.to_csv(df_filepath, index=False)
    print(f"Processed data saved to {df_filepath}")

def preprocess_pipeline(filepath, return_format='dataframe'):
    """Complete preprocessing pipeline with format options."""
    df = load_and_initial_clean(filepath)
    df = clean_rename_and_select_columns(df)
    df = apply_custom_mappings(df)
    df_final, scaler = scale_features(df)

    # Generate a unique ID for the output file
    unique_id = uuid.uuid4().hex[:8]
    output_filepath = f"processed_students_{unique_id}.csv"
    save_results(df_final, output_filepath)

    if return_format == 'api':
        students_list = convert_to_api_format(df_final)
        return students_list, scaler
    else:
        return df_final, scaler

if __name__ == "__main__":
    try:
        dummy_data = {
            'Student ID': ['S001', 'S002', 'S003', 'S004', 'S005'],
            'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'Program & Year Level': ['BSCS-4', 'BSIT-2', 'BSECE-3', '', 'BSME-1'], # Added empty string case
            'How do you feel your academic performance has changed this year?': ['Slightly Improved', 'Stayed the Same', 'Significantly Declined', 'Slightly Improved', 'Stayed the Same'],
            'How would you rate your current academic workload?': ['Moderate', 'Challenging', 'Very Challenging', 'Easy', 'Moderate'],
            'Gender': ['Female', 'Male', 'Female', 'Male', 'Female'],
            'How do you prefer to learn? (Select all that apply)”': ['Visual', 'Auditory,Kinesthetic', 'Reading/Writing', 'Visual,Auditory', 'Kinesthetic'],
            'How do you usually handle academic challenges?': ['Asks after trying for a while', 'Prefers to solve problems alone', 'Waits too long to ask', 'Always asks for help immediately', 'Asks after trying for a while'],
            'Which personality type best describes you?': ['Introvert', 'Extrovert', 'Ambivert', 'Introvert', 'Extrovert'],
            'What activities or hobbies do you enjoy outside of class? (Select/Write at least 3)': ['Reading,Gaming,Drawing', 'Sports,Music', 'Cooking,Hiking', 'Photography', 'Painting,Writing,Music'],
            'How would you describe your current financial situation as a student?': ['Comfortable', 'Struggling', 'Very Comfortable', 'Somewhat Challenging', 'Comfortable'],
            'What is your birth order among your siblings?': ['Oldest', 'Youngest', 'Middle Child', 'Only Child', 'Oldest'],
            'Do you have major responsibilities outside of school (e.g., part-time job, caregiving)? If yes, please describe briefly. Leave blank if none.': ['None', 'Part-time job', '', 'Caregiving', 'None'],
            'Grade_Subject1': ['1.0 - 1.2', '2.1 - 2.5', '2.51 - 3.0', '1.21 - 1.4', '1.0 - 1.2'],
            'Grade_Subject2': ['1.41 - 2.0', '2.51 - 3.0', '1.0 - 1.2', '2.1 - 2.5', '1.41 - 2.0'],
            **{f'Grade_Dummy{i}': ['1.0 - 1.2'] * 5 for i in range(3, 75)}
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_filepath = 'dummy_raw_data.csv'
        dummy_df.to_csv(dummy_filepath, index=False)
        print(f"Created dummy CSV at: {dummy_filepath}")

        processed_data, scaler = preprocess_pipeline(dummy_filepath, return_format='api')

        print("\nPreprocessing complete. Data is ready.")
        if processed_data:
            print(f"First 3 students example (API format):")
            for i, student in enumerate(processed_data[:3]):
                print(f"Student {i+1}: {student}")
                if 'Program and Year' in student:
                    print(f"  Program and Year: {student['Program and Year']}")
                else:
                    print("  'Program and Year' not found in processed data.")
        else:
            print("No data processed.")

        os.remove(dummy_filepath)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
