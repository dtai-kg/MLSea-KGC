import pandas as pd
import config
import json
from preprocessing_modules import *
from json import dumps, loads

def get_csv_updates(csv1, csv2, columns):

    # Load CSV files into DataFrames
    df1 = pd.read_csv(csv1)[columns]
    df2 = pd.read_csv(csv2)[columns]

    # Find rows in df2 that are not in df1
    updates = pd.merge(df1, df2, how='outer', indicator=True).query('_merge == "right_only"').drop('_merge', axis=1)
    return updates

def get_json_updates(json1, json2, output_json):

    # Load JSON files
    with open(json1, 'r') as file:
        data1 = json.load(file)

    with open(json2, 'r') as file:
        data2 = json.load(file)

    # Identify new items in the updated array
    set1 = set(dumps(x, sort_keys=True) for x in data1)
    set2 = set(dumps(x, sort_keys=True) for x in data2)
    new_items = [loads(x) for x in set2.difference(set1)]

    with open(output_json, 'w') as new_file:
        json.dump(new_items, new_file, indent=2)

# Example usage
# get_json_updates(config.PWC_INPUT + config.ORIGINAL_DATA_FOLDER + "paper_code_links.json",
#  config.PWC_INPUT + config.UPDATE_MONTH_FOLDER + "paper_code_links.json", 
#  config.PWC_INPUT + config.UPDATE_MONTH_FOLDER + "paper_code_links2.json")

