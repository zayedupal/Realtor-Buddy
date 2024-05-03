import pandas as pd
import json
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime, timezone
from dotenv import load_dotenv

from data_processing.postgres_service import get_connection

load_dotenv()

def read_json_from_folder(folder_path):
    """Reads all files within a folder and its subdirectories.

    Args:
    folder_path: The path to the folder containing the files.
    """
    file_paths = []
    for root, directories, files in os.walk(folder_path):
        for filename in files:
            if '.json' not in filename:
                continue
            # Construct the full path of the file
            file_path = os.path.join(root, filename)
            file_paths.append(file_path)

    return file_paths


def infer_sql_type(value):
    if isinstance(value, np.bool_):
        return 'BOOLEAN'
    elif isinstance(value, np.int64):
        return 'BIGINT'
    elif isinstance(value, np.integer):
        return 'INTEGER'
    elif isinstance(value, np.floating):
        return 'REAL'
    # elif isinstance(value, str):
    #     return 'TEXT'
    else:
        return 'TEXT'


def convert_to_native_type(val):
    """
    Convert all potential numpy types to native Python types before insertion to SQL.
    """
    # if pd.isna(val):  # Handle missing values
    #     return None

    # Ensure val is not an iterable like a list or numpy array
    # This part depends on whether your database scheme expects any lists or arrays to be inserted
    if isinstance(val, (list, np.ndarray)):
        # Convert list elements to string if necessary, or process differently as needed
        return str(val)  # Simple solution: convert lists or arrays to string

    # Direct conversion for individual numpy types to native Python types
    if isinstance(val, np.integer):
        return int(val)
    elif isinstance(val, np.floating):
        return float(val)
    elif isinstance(val, np.bool_):
        return bool(val)
    elif isinstance(val, (np.str_, str)):
        return str(val)

    return val  # Fallback if no types matched, return as is (e.g., for Python native types or unrecognized types)


def create_table(json, json_features, connection, table_name, primary_key):
    row = json[0]
    selected_data = {key: row[key] for key in json_features if key in row}
    df = pd.json_normalize(selected_data, sep='_')
    df[primary_key] = df[primary_key].astype(int)
    cursor = connection.cursor()
    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ("
    for column in df.columns:
        sample_value = df.at[0, column] if not df[column].empty else None
        sql_type = infer_sql_type(sample_value)
        create_table_query += f"{column} {sql_type} PRIMARY KEY, " if column == primary_key else f"{column} {sql_type}, "
    create_table_query += (f"create_date TIMESTAMP WITH TIME ZONE NOT NULL, "
                           f"update_date TIMESTAMP WITH TIME ZONE NOT NULL);")

    cursor.execute(create_table_query)
    connection.commit()
    return df.columns.values


def upsert_in_table(cur_row, json_features, connection, table_name, primary_key, table_columns):
    selected_data = {key: cur_row[key] for key in json_features if key in cur_row}
    df = pd.json_normalize(selected_data, sep='_')
    df = df.drop(columns=[col for col in df.columns if col not in table_columns])
    df[primary_key] = df[primary_key].astype(int)
    df['create_date'] = datetime.now(timezone.utc)
    df['update_date'] = datetime.now(timezone.utc)
    cursor = connection.cursor()
    columns_string = ", ".join(df.columns)
    values_string = ", ".join(["%s" for _ in df.columns])
    update_clause = ', '.join(
        [f"{col} = EXCLUDED.{col}" for col in df.columns if col not in [primary_key, 'create_date']])
    insert_query = f"INSERT INTO {table_name} ({columns_string}) VALUES ({values_string}) "
    insert_query += f"ON CONFLICT ({primary_key}) DO UPDATE SET {update_clause}"

    # Example usage during insertion
    insert_values = tuple(convert_to_native_type(val) for val in df.loc[0].values)
    cursor.execute(insert_query, insert_values)
    connection.commit()


if __name__ == '__main__':
    json_data_folder = "data/georgia top cities"
    # json_data_folder = "data/others"
    json_features = pd.read_csv('json_features.csv').columns.values
    table_name = "property"
    primary_key = 'zpid'

    files = read_json_from_folder(json_data_folder)

    for i, file in tqdm(enumerate(files)):
        with open(file, 'r') as f:
            data = json.load(f)

        if not data or 'data' not in data:
            print(f'no data in the json file: {file}')
            continue

        json_data = data['data']
        if len(json_data) <= 0:
            print("empty json data")
            exit()

        # connect with DB
        connection = get_connection()
        # create DB table in first iteration if table doesn't exist
        if i == 0:
            table_columns = create_table(json_data, json_features, connection, table_name, primary_key)

        for cur_row in json_data:
            upsert_in_table(cur_row, json_features, connection, table_name, primary_key, table_columns)

    connection.cursor().close()
    connection.close()
