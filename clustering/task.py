import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


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


def process_grades(df):
    """Discretize grades and calculate averages."""
    grade_cols_1 = df.columns[3:58]
    grade_cols_2 = df.columns[68:74]
    grade_cols = grade_cols_1.union(grade_cols_2)

    df_grade_cols = df[grade_cols]

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

    df_result = df.copy()
    df_result['Average'] = df_grade_cols['Average']

    return df_result


def clean_and_transform(df):
    """Clean data and transform categorical features."""
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    df.replace('', np.nan, inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('Unknown')

    performance_map = {
        'Significantly Improved': 2, 'Slightly Improved': 1, 'Stayed the Same': 0,
        'Slightly Declined': -1, 'Significantly Declined': -2
    }
    df['Academic Performance Change'] = df['How do you feel your academic performance has changed this year?'].map(performance_map).fillna(0)

    workload_map = {
        'Very Easy': 1, 'Easy': 2, 'Moderate': 3, 'Challenging': 4, 'Very Challenging': 5
    }
    df['Workload Rating'] = df['How would you rate your current academic workload?'].map(workload_map).fillna(0)

    df['Gender'] = df['Gender'].astype(str).apply(lambda x: 1 if x.lower() == 'female' else (0 if x.lower() == 'male' else 2))

    learning_styles = ['Visual', 'Auditory', 'Reading/Writing', 'Kinesthetic']
    for style in learning_styles:
        source_col = 'How do you prefer to learn? (Select all that apply)”'
        if source_col in df.columns:
            df[f'Learning_{style}'] = df[source_col].astype(str).str.contains(style, na=False).astype(int)
        else:
            df[f'Learning_{style}'] = 0

    help_map = {
        'Always asks for help immediately': 0, 'Asks after trying for a while': 1,
        'Waits too long to ask': 2, 'Prefers to solve problems alone': 3
    }
    df['Help Seeking'] = df['How do you usually handle academic challenges?'].map(help_map).fillna(0)

    if 'Which personality type best describes you?' in df.columns:
        df['Personality_Raw'] = df['Which personality type best describes you?'].astype(str).str.extract(r'(Introvert|Ambivert|Extrovert)')[0]
        df['Personality_Raw'] = df['Personality_Raw'].fillna('Unknown')
    else:
        df['Personality_Raw'] = 'Unknown'

    if 'What activities or hobbies do you enjoy outside of class? (Select/Write at least 3)' in df.columns:
        df['Hobby Count'] = df['What activities or hobbies do you enjoy outside of class? (Select/Write at least 3)'].astype(str).str.split(',').apply(lambda x: len(x) if isinstance(x, list) else 0)
    else:
        df['Hobby Count'] = 0

    financial_map = {
        'Very Comfortable': 4, 'Comfortable': 3, 'Somewhat Challenging': 2,
        'Struggling': 1, 'Severely Struggling': 0
    }
    df['Financial Status'] = df['How would you describe your current financial situation as a student?'].map(financial_map).fillna(0)

    marital_col = None
    for col in df.columns:
        if 'relationship status' in col.lower():
            marital_col = col
            break

    marital_map = {
        'Together': 'Together', 'Separated': 'Separated',
        'One or Both Parents Deceased': 'Deceased', 'Prefer not to say': 'Unknown'
    }
    if marital_col and marital_col in df.columns:
        df['Parents Marital Status_Raw'] = df[marital_col].map(marital_map)
        df['Parents Marital Status_Raw'] = df['Parents Marital Status_Raw'].fillna('Unknown')
    else:
        df['Parents Marital Status_Raw'] = 'Unknown'

    birth_order_map = {
        'Oldest': 'Oldest', 'Middle Child': 'Middle',
        'Youngest': 'Youngest', 'Only Child': 'Only'
    }
    df['Birth Order_Raw'] = df['What is your birth order among your siblings?'].map(birth_order_map)
    df['Birth Order_Raw'] = df['Birth Order_Raw'].fillna('Unknown')

    resp_col = 'Do you have major responsibilities outside of school (e.g., part-time job, caregiving)? If yes, please describe briefly. Leave blank if none.'
    if resp_col in df.columns:
        df['Has External Responsibilities'] = df[resp_col].astype(str).apply(
            lambda x: 0 if x.lower() in ['unknown', 'nan', '', 'none'] else 1
        )
    else:
        df['Has External Responsibilities'] = 0

    return df


def select_final_features(df):
    """Select and process final features for modeling."""
    # IMPORTANT: Keep original identifiers and program/year for Student model creation
    # These columns should NOT be encoded or scaled if they are only for identification/display
    identifier_columns = ['Student ID', 'Name', 'program_str', 'year_int']

    # Features that WILL be encoded/scaled for clustering
    features_for_clustering_raw = [
        'Academic Performance Change', 'Workload Rating',
        'Learning_Visual', 'Learning_Auditory', 'Learning_Reading/Writing', 'Learning_Kinesthetic',
        'Help Seeking', 'Personality_Raw', 'Hobby Count',
        'Financial Status', 'Parents Marital Status_Raw', 'Birth Order_Raw',
        'Has External Responsibilities', 'Average', 'Gender'
    ]

    # Combine all columns needed in the final DataFrame
    all_needed_columns = identifier_columns + [col for col in features_for_clustering_raw if col not in identifier_columns]

    # Only include columns that actually exist in the DataFrame
    existing_columns = [col for col in all_needed_columns if col in df.columns]
    df_final = df[existing_columns].copy()

    # Fill NaN values with appropriate defaults for numeric features
    numeric_fill_cols = [
        'Academic Performance Change', 'Workload Rating', 'Help Seeking',
        'Hobby Count', 'Financial Status', 'Average', 'Gender',
        'Learning_Visual', 'Learning_Auditory', 'Learning_Reading/Writing',
        'Learning_Kinesthetic', 'Has External Responsibilities'
    ]
    for col in numeric_fill_cols:
        if col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0.0)

    # Categorical features that need encoding for clustering
    categorical_features_to_encode = ['Personality_Raw', 'Birth Order_Raw'] # program_str is NOT here
    label_encoders = {}
    for col_raw in categorical_features_to_encode:
        if col_raw in df_final.columns:
            df_final[col_raw] = df_final[col_raw].astype(str).fillna('Unknown') # Ensure string type
            le = LabelEncoder()
            # Create a new column for the encoded feature (e.g., 'Personality' from 'Personality_Raw')
            df_final[col_raw.replace('_Raw', '')] = le.fit_transform(df_final[col_raw])
            label_encoders[col_raw] = le

    # One-hot encode marital status
    if 'Parents Marital Status_Raw' in df_final.columns:
        df_final['Parents Marital Status_Raw'] = df_final['Parents Marital Status_Raw'].astype(str).fillna('Unknown')
        df_final['Marital_Separated'] = (df_final['Parents Marital Status_Raw'] == 'Separated').astype(int)
        df_final['Marital_Together'] = (df_final['Parents Marital Status_Raw'] == 'Together').astype(int)
    else:
        df_final['Marital_Separated'] = 0
        df_final['Marital_Together'] = 0

    # Define the final list of features that the clustering model expects
    # These are the *encoded/processed* versions of the features
    features_for_scaling_and_clustering = [
        'Academic Performance Change', 'Workload Rating',
        'Learning_Visual', 'Learning_Auditory', 'Learning_Reading/Writing', 'Learning_Kinesthetic',
        'Help Seeking', 'Personality', 'Hobby Count', # 'Personality' is now the encoded value
        'Financial Status', 'Birth Order', # 'Birth Order' is now the encoded value
        'Has External Responsibilities', 'Average', 'Gender',
        'Marital_Separated', 'Marital_Together'
    ]

    # Filter to only include columns that are actually in df_final and are numeric for scaling
    existing_features_for_scaling = [col for col in features_for_scaling_and_clustering if col in df_final.columns and pd.api.types.is_numeric_dtype(df_final[col])]

    if existing_features_for_scaling:
        for col in existing_features_for_scaling:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0.0)
            df_final[col] = df_final[col].replace([np.inf, -np.inf], 0.0)

        scaler = StandardScaler()
        try:
            df_final[existing_features_for_scaling] = scaler.fit_transform(df_final[existing_features_for_scaling])
        except Exception as e:
            print(f"DEBUG: Scaling error: {e}. Attempting manual normalization.")
            for col in existing_features_for_scaling:
                mean_val = df_final[col].mean()
                std_val = df_final[col].std()
                if std_val > 0:
                    df_final[col] = (df_final[col] - mean_val) / std_val
                else:
                    df_final[col] = 0.0
            scaler = None
    else:
        scaler = StandardScaler()

    df_final = df_final.fillna(0.0)
    df_final = df_final.replace([np.inf, -np.inf], 0.0)

    print(f"DEBUG (task.py - select_final_features): Final dataframe shape: {df_final.shape}")
    print(f"DEBUG (task.py - select_final_features): Final columns: {list(df_final.columns)}")
    # Debug print to ensure program_str and year_int are still present and their original values
    print(f"DEBUG (task.py - select_final_features): Sample program_str (before API format): {df_final['program_str'].head().tolist()}")
    print(f"DEBUG (task.py - select_final_features): Sample year_int (before API format): {df_final['year_int'].head().tolist()}")


    return df_final, label_encoders, scaler

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

        processed_data, label_encoders, scaler = preprocess_pipeline(dummy_filepath, return_format='api')

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
