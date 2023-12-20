# Data Paths
INPUT_PATH = "/Users/ioannisdasoulas/Desktop/ML-Discovery/ML-KG/Data/"
OPENML_INPUT = "/Users/ioannisdasoulas/Desktop/ML-Discovery/ML-KG/Data/OpenML-Data/"
KAGGLE_INPUT = "/Users/ioannisdasoulas/Desktop/ML-Discovery/ML-KG/Data/Kaggle-Data/"
PWC_INPUT = "/Users/ioannisdasoulas/Desktop/ML-Discovery/ML-KG/Data/PwC-Data/"
OUTPUT_PATH = "/Users/ioannisdasoulas/Desktop/ML-Discovery/ML-KG/RDF_Dumps/"
ORIGINAL_DATA_FOLDER = "Original-Data/"
#UPDATE_MONTH_FOLDER = "December2023/"

# OpenML API Checkpoints
OPENML_RUN_CHECKPOINT = 4037070
OPENML_RUN_CURRENT_OFFSET = 6000000
OPENML_DATASET_CHECKPOINT = 5399
OPENML_FLOW_CHECKPOINT = 47250
OPENML_TASK_CHECKPOINT = 16736

def update_openml_checkpoints(run_cp, dataset_cp, task_cp, flow_cp):

    # Open the constants.py file for editing
    with open('config.py', 'r') as file:
        content = file.read()

    # Update the values in memory
    content = content.replace('OPENML_RUN_CHECKPOINT = 4037070', 'OPENML_RUN_CHECKPOINT = ' + str(run_cp))
    content = content.replace('OPENML_DATASET_CHECKPOINT = 5399', 'OPENML_DATASET_CHECKPOINT = ' + str(dataset_cp))
    content = content.replace('OPENML_FLOW_CHECKPOINT = 47250', 'OPENML_FLOW_CHECKPOINT = ' + str(task_cp))
    content = content.replace('OPENML_TASK_CHECKPOINT = 16736', 'OPENML_TASK_CHECKPOINT = ' + str(flow_cp))

    # Write the changes back to the constants.py file
    with open('config.py', 'w') as file:
        file.write(content)

    return