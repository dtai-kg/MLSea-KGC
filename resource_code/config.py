# Data Paths
INPUT_PATH = "/Users/ioannisdasoulas/Desktop/ML-Discovery/ML-KG/Data/"
OPENML_INPUT = "/Users/ioannisdasoulas/Desktop/ML-Discovery/\
                ML-KG/Data/OpenML-Data/"
KAGGLE_INPUT = "/Users/ioannisdasoulas/Desktop/ML-Discovery/\
                ML-KG/Data/Kaggle-Data/"
PWC_INPUT = "/Users/ioannisdasoulas/Desktop/ML-Discovery/ML-KG/Data/PwC-Data/"
OUTPUT_PATH = "/Users/ioannisdasoulas/Desktop/ML-Discovery/ML-KG/RDF_Dumps/"
ORIGINAL_DATA_FOLDER = "23-05-2024/"
UPDATE_MONTH_FOLDER = "11-12-2025/"

# OpenML API Checkpoints
OPENML_RUN_CHECKPOINT = 4037963
OPENML_RUN_CURRENT_OFFSET = 6000000
OPENML_DATASET_CHECKPOINT = 5482
OPENML_FLOW_CHECKPOINT = 16851
OPENML_TASK_CHECKPOINT = 47273

# Dumps current file number
OPENML_TASK_DUMP_PART = 2
OPENML_FLOW_DUMP_PART = 2
OPENML_DATASET_DUMP_PART = 2
OPENML_RUN_DUMP_PART = 30
KAGGLE_DUMP_PART = 2
PWC_DUMP_PART = 2

# Triples limit per dump
OPENML_DUMP_LIMIT = 50000000
KAGGLE_DUMP_LIMIT = 30000000
PWC_DUMP_LIMIT = 20000000


def update_openml_checkpoints(run_cp, dataset_cp, task_cp, flow_cp):

    # Open the constants.py file for editing
    with open('/Users/ioannisdasoulas/Desktop/ML-Discovery/\
              mlsea-discover/resource_code/config.py', 'r') as file:
        content = file.read()

    # Update the values in memory
    content = content.replace(
        f'OPENML_RUN_CHECKPOINT = {OPENML_RUN_CHECKPOINT}',
        'OPENML_RUN_CHECKPOINT = ' + str(run_cp))
    content = content.replace(
        f'OPENML_DATASET_CHECKPOINT = {OPENML_DATASET_CHECKPOINT}',
        'OPENML_DATASET_CHECKPOINT = ' + str(dataset_cp))
    content = content.replace(
        f'OPENML_FLOW_CHECKPOINT = {OPENML_FLOW_CHECKPOINT}',
        'OPENML_FLOW_CHECKPOINT = ' + str(flow_cp))
    content = content.replace(
        f'OPENML_TASK_CHECKPOINT = {OPENML_TASK_CHECKPOINT}',
        'OPENML_TASK_CHECKPOINT = ' + str(task_cp))

    # Write the changes back to the constants.py file
    with open('config.py', 'w') as file:
        file.write(content)

    return
