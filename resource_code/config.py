# Data Paths
INPUT_PATH = "/Users/ioannisdasoulas/Desktop/ML-Discovery/ML-KG/Data/"
OPENML_INPUT = "/Users/ioannisdasoulas/Desktop/ML-Discovery/ML-KG/Data/OpenML-Data/"
KAGGLE_INPUT = "/Users/ioannisdasoulas/Desktop/ML-Discovery/ML-KG/Data/Kaggle-Data/"
PWC_INPUT = "/Users/ioannisdasoulas/Desktop/ML-Discovery/ML-KG/Data/PwC-Data/"
OUTPUT_PATH = "/Users/ioannisdasoulas/Desktop/ML-Discovery/ML-KG/RDF_Dumps/"
ORIGINAL_DATA_FOLDER = "Original-Data/"
UPDATE_MONTH_FOLDER = "10-01-2024/"

# OpenML API Checkpoints
OPENML_RUN_CHECKPOINT = 4037082
OPENML_RUN_CURRENT_OFFSET = 6000000
OPENML_DATASET_CHECKPOINT = 5402
OPENML_FLOW_CHECKPOINT = 16751
OPENML_TASK_CHECKPOINT = 47250

# Dumps current file number
OPENML_TASK_DUMP_PART = 1 
OPENML_FLOW_DUMP_PART = 1
OPENML_DATASET_DUMP_PART = 1
OPENML_RUN_DUMP_PART = 29
KAGGLE_DUMP_PART = 1
PWC_DUMP_PART = 1

# Triples limit per dump
OPENML_DUMP_LIMIT = 50000000
KAGGLE_DUMP_LIMIT = 30000000
PWC_DUMP_LIMIT = 20000000

def update_openml_checkpoints(run_cp, dataset_cp, task_cp, flow_cp):

    # Open the constants.py file for editing
    with open('config.py', 'r') as file:
        content = file.read()

    # Update the values in memory
    content = content.replace('OPENML_RUN_CHECKPOINT = 4037082', 'OPENML_RUN_CHECKPOINT = ' + str(run_cp))
    content = content.replace('OPENML_DATASET_CHECKPOINT = 5402', 'OPENML_DATASET_CHECKPOINT = ' + str(dataset_cp))
    content = content.replace('OPENML_FLOW_CHECKPOINT = 16751', 'OPENML_FLOW_CHECKPOINT = ' + str(flow_cp))
    content = content.replace('OPENML_TASK_CHECKPOINT = 47250', 'OPENML_TASK_CHECKPOINT = ' + str(task_cp))

    # Write the changes back to the constants.py file
    with open('config.py', 'w') as file:
        file.write(content)

    return