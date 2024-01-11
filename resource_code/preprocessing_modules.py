import re
import json
from datetime import datetime
import numpy as np
import pandas as pd

def preprocess_json_strings(data, preprocess_string_func, preprocess_datetime_func):

    if isinstance(data, dict):
        for key, value in data.items():
            if key.lower() == "date" or key.lower() == "paper_date" or key.lower() == "introduced_date":
                data[key] = preprocess_datetime_func(value) if value is not None else None
            elif key.lower() == "homepage":
                data[key] = value.lower() if value is not None else None
            else:
                data[key] = preprocess_json_strings(value, preprocess_string_func, preprocess_datetime_func)
    
    elif isinstance(data, list):
        for i, item in enumerate(data): 
            data[i] = preprocess_json_strings(item, preprocess_string_func, preprocess_datetime_func)

    elif isinstance(data, str):
        data = preprocess_string_func(data)

    return data


def preprocess_string(input_string):
    # Replace double quotes with single quotes
    step1 = input_string.replace('"', "'")

    # Replace newlines with spaces
    step2 = step1.replace('\n', ' ')

    # Replace newline characters (\n), tab characters (\t), or carriage return (\r) with spaces
    step3 = re.sub(r'[\n\t\r\b\v\0]', ' ', step2)

    # Replace non-escaped backslashes with spaces
    step4 = re.sub(r'(?<!\\)\\(?!\\)', ' ', step3)

    # Replace consecutive whitespaces with a single space
    step5 = re.sub(r'\s+', ' ', step4)

    return step5

def preprocess_datetime(input_datetime_str):

    if input_datetime_str is np.nan:
        return np.nan
    if input_datetime_str is None:
        return None
    if input_datetime_str == "":
        return ""

    # Attempt to detect the input format
    formats_to_try = ["%d/%m/%Y %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m-%d-%Y %H:%M:%S",
    "%d-%m-%Y", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"]

    for date_format in formats_to_try:
        try:
            # Try to parse the input string to a datetime object
            dt_object = datetime.strptime(input_datetime_str, date_format)

            # If successful, check if only the date is provided and set the time to 00:00:00
            if date_format in ["%m/%d/%Y", "%Y/%m/%d", "%d/%m/%Y",
            "%m-%d-%Y", "%Y-%m-%d", "%d-%m-%Y"]:
                dt_object = dt_object.replace(hour=0, minute=0, second=0)

            # Format the datetime object to the desired output format
            output_format = "%Y-%m-%dT%H:%M:%S"
            output_datetime_str = dt_object.strftime(output_format)
            return output_datetime_str
        except ValueError:
            pass  # Continue to the next format if the current one fails

    # If none of the formats work, raise an error or handle it as appropriate for your use case
    raise ValueError("Unsupported datetime format: {}".format(input_datetime_str))

def preprocess_df_strings(df):

    df = df.applymap(lambda x: preprocess_string(x) if isinstance(x, str) else x)
    return df

def preprocess_json(path, filename):

    with open(path + filename, 'r') as file:
        json_data = json.load(file)

    processed_data = preprocess_json_strings(json_data, preprocess_string, preprocess_datetime)

    with open(path + filename, 'w') as file:
        json.dump(processed_data, file, indent=2)

def pre_process_pwc_evaluations(datapath):

    json_file_path = "evaluations.json"
    with open(datapath+json_file_path, 'r') as j:
        contents = json.loads(j.read())

    paper_titles = []
    model_names = []
    datasets = []
    metrics = []
    values = []

    for task in contents:
        for dataset in task["datasets"]:
            for row in dataset["sota"]["rows"]:
                for metric in row["metrics"]:
                    paper_titles.append(row["paper_title"])
                    model_names.append(row["model_name"])
                    datasets.append(dataset["dataset"])
                    metrics.append(metric)
                    values.append(row["metrics"][metric])
            

    for content in contents:
        for task in content["subtasks"]:
            for dataset in task["datasets"]:
                for row in dataset["sota"]["rows"]:
                    for metric in row["metrics"]:
                        paper_titles.append(row["paper_title"])
                        model_names.append(row["model_name"])
                        datasets.append(dataset["dataset"])
                        metrics.append(metric)
                        values.append(row["metrics"][metric])
                
                

    for content in contents:
        for content2 in content["subtasks"]:
            for task in content2["subtasks"]:
                for task in content["subtasks"]:
                    for dataset in task["datasets"]:
                        for row in dataset["sota"]["rows"]:
                            for metric in row["metrics"]:
                                paper_titles.append(row["paper_title"])
                                model_names.append(row["model_name"])
                                datasets.append(dataset["dataset"])
                                metrics.append(metric)
                                values.append(row["metrics"][metric])
                        
                        
    evaluations_df = pd.DataFrame({"paper_title" : paper_titles,
                      "model_name": model_names,
                      "dataset": datasets,
                      "metric_name": metrics,
                      "metric_value": values})

    evaluations_df.to_csv(datapath + "evaluations.csv", index=False)

    return



