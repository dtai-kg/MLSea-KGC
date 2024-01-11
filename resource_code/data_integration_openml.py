import morph_kgc
from get_data_sample import get_task_batch, get_dataset_batch, get_run_batch, get_flow_batch
from queries import *
import sys
import pandas as pd
import warnings
from dump_storage import * 
from preprocessing_modules import *
import config
from openml_data_collector import get_checkpoints
warnings.simplefilter(action='ignore', category=FutureWarning)


def integrate_openml_tasks(tasks_df, targetpath, files, file_part, file_subpart):
    
    tasks_df = preprocess_df_strings(tasks_df)
    data_dict = {"tasks_df": tasks_df}
    graph = morph_kgc.materialize('./morph_config/task_conf.ini', data_dict)
    print("RDF generated!")
    print("Saving to disk...")
    # for s,p,o in graph.triples((None, None, None)):
    #     print(s,p,o)

    filename = "openml_tasks_" + str(file_part) + "_part_" + str(file_subpart) + ".nt"
    graph.serialize(destination = targetpath + filename, format = "nt", encoding = "utf-8")
    files.append(filename)
    

    return files, len(graph)


def integrate_openml_datasets(
    datasets_df, dataset_creators_df, dataset_tags_df,
    dataset_features_df, dataset_references_df, targetpath, 
    files, file_part, file_subpart):

    datasets_df = preprocess_df_strings(datasets_df)
    dataset_creators_df = preprocess_df_strings(dataset_creators_df)
    dataset_tags_df = preprocess_df_strings(dataset_tags_df)
    dataset_features_df = preprocess_df_strings(dataset_features_df)
    data_dict = {"datasets_df": datasets_df,
                "dataset_creators_df": dataset_creators_df,
                "dataset_tags_df": dataset_tags_df,
                "dataset_features_df": dataset_features_df,
                "dataset_references_df": dataset_references_df}
    graph = morph_kgc.materialize('./morph_config/dataset_conf.ini', data_dict)
    print("RDF generated!")
    print("Saving to disk...")
    # for s,p,o in graph.triples((None, None, None)):
    #     print(s,p,o)

    filename = "openml_datasets_" + str(file_part) + "_part_" + str(file_subpart) + ".nt"
    graph.serialize(destination = targetpath + filename, format = "nt", encoding = "utf-8")
    files.append(filename)
    

    return files, len(graph)


def integrate_openml_runs(
    runs_df, run_evaluations_df, run_settings_df, targetpath, 
    files, file_part, file_subpart):

    runs_df['upload_time'] = runs_df['upload_time'].apply(preprocess_datetime)
    runs_df = preprocess_df_strings(runs_df)
    run_settings_df = preprocess_df_strings(run_settings_df)
    run_evaluations_df = preprocess_df_strings(run_evaluations_df)
    data_dict = {"runs_df": runs_df,
                "run_evaluations_df": run_evaluations_df,
                "run_settings_df": run_settings_df}
    graph = morph_kgc.materialize('./morph_config/run_conf.ini', data_dict)
    print("RDF generated!")
    print("Saving to disk...")
    # for s,p,o in graph.triples((None, None, None)):
    #     print(s,p,o)

    filename = "openml_runs_" + str(file_part) + "_part_" + str(file_subpart) + ".nt"
    graph.serialize(destination = targetpath + filename, format = "nt", encoding = "utf-8")
    files.append(filename)
    

    return files, len(graph)


def integrate_openml_flows(
    flows_df, flow_params_df, flow_tags_df, flow_dependencies_df, targetpath, 
    files, file_part, file_subpart):

    flows_df = preprocess_df_strings(flows_df)
    flow_params_df = preprocess_df_strings(flow_params_df)
    flow_dependencies_df = preprocess_df_strings(flow_dependencies_df)
    data_dict = {"flows_df": flows_df,
                "flow_params_df": flow_params_df,
                "flow_tags_df": flow_tags_df,
                "flow_dependencies_df": flow_dependencies_df}
    graph = morph_kgc.materialize('./morph_config/flow_conf.ini', data_dict)
    print("RDF generated!")
    print("Saving to disk...")
    # for s,p,o in graph.triples((None, None, None)):
    #     print(s,p,o)

    filename = "openml_flows_" + str(file_part) + "_part_" + str(file_subpart) + ".nt"
    graph.serialize(destination = targetpath + filename, format = "nt", encoding = "utf-8")
    files.append(filename)
    

    return files, len(graph)

def find_instance_count(db, graph):

    query = get_query(SELECT_ID_COUNT, graph)
    try:
        count = int(db.get_triples(query)["results"]["bindings"][0]["instanceCount"]["value"])
    except:
        print("Connection problem! Returning...")
        sys.exit()

    return count

# Direct integration implementation
# def integrate_openml_flows(
#     flows_df, flow_params_df, flow_tags_df, flow_dependencies_df, db, named_graph): ## Integrate directly to db

#     data_dict = {"flows_df": flows_df,
#                 "flow_params_df": flow_params_df,
#                 "flow_tags_df": flow_tags_df,
#                 "flow_dependencies_df": flow_dependencies_df}
#     graph = morph_kgc.materialize('./morph_config/flow_conf.ini', data_dict)
#     print("RDF generated! Uploading to database...")
#     db.insert_data(named_graph, graph)

#     return 

# def integrate_openml_tasks_from_csv(datapath, db, batch_size): ## Integration directly to db

#     named_graph = OPENML_TASK_GRAPH

#     task_batch_size, task_batch_offset = batch_size, find_instance_count(db, named_graph)
#     tasks_df, tasks_clearance = get_task_batch(
#     datapath, task_batch_offset, task_batch_size)

#     while tasks_clearance == True:
        
#         print(f"\nIntegrating triples from Task {task_batch_offset + 1} to Task {task_batch_size+task_batch_offset}...")
#         integrate_openml_tasks(tasks_df, db, named_graph)
#         print("Integration complete!\n")

#         task_batch_offset += task_batch_size

#         tasks_df, tasks_clearance = get_task_batch(
#         datapath, task_batch_offset, task_batch_size)

#     print("No more task data to integrate. Returning...\n")

#     return 

def integrate_openml_tasks_from_csv(datapath, targetpath, batch_offset, batch_size, update=False):

    print("Preparing for OpenML Task integration...\n")
    tasks_df, tasks_clearance = get_task_batch(
    datapath, batch_offset, batch_size)

    files = []
    if update == False: 
        file_subpart = 1
        file_part = 1
        total_triples = 0
    elif update == True and tasks_clearance == True: 
        file_subpart = 2
        file_part = config.OPENML_TASK_DUMP_PART
        current_dump = "openml_tasks_" + str(file_part) + ".nt"
        total_triples = count_dump_triples(targetpath + current_dump + ".gz")
        unzip_and_save(targetpath + current_dump + ".gz")
        files = [current_dump]   
    
    goal_triples = config.OPENML_DUMP_LIMIT

    while tasks_clearance == True:
        
        print(f"\nIntegrating triples from Task {batch_offset + 1} to Task {batch_size+batch_offset}...")
        files, n_triples = integrate_openml_tasks(tasks_df, targetpath, files, file_part, file_subpart)
        total_triples += n_triples
        print("Integration complete!")
        print("Current dump triple count:", total_triples, "\n")
        file_subpart += 1

        if total_triples > goal_triples:
            output_file = "openml_tasks_" + str(file_part) + ".nt.gz"
            concatenate_and_compress(targetpath, files, output_file)
            delete_files(targetpath, files)
            file_subpart = 1
            file_part += 1
            total_triples = 0
            files = []

        batch_offset += batch_size

        tasks_df, tasks_clearance = get_task_batch(
        datapath, batch_offset, batch_size)

    if len(files) > 0:
        output_file = "openml_tasks_" + str(file_part) + ".nt.gz"
        concatenate_and_compress(targetpath, files, output_file)
        delete_files(targetpath, files)

    print("No more task data to integrate. Returning...\n")

    return 

def integrate_openml_flows_from_csv(datapath, targetpath, batch_offset, batch_size, update=False):

    print("Preparing for OpenML Flow integration...\n")
    flows_df, flow_params_df, flow_tags_df, flow_dependencies_df, flow_clearance = get_flow_batch(
    datapath, batch_offset, batch_size)
    
    files = []
    if update == False: 
        file_subpart = 1
        file_part = 1
        total_triples = 0
    elif update == True and flow_clearance == True: 
        file_subpart = 2
        file_part = config.OPENML_FLOW_DUMP_PART
        current_dump = "openml_flows_" + str(file_part) + ".nt"
        total_triples = count_dump_triples(targetpath + current_dump + ".gz")
        unzip_and_save(targetpath + current_dump + ".gz")
        files = [current_dump]   
    
    goal_triples = config.OPENML_DUMP_LIMIT

    while flow_clearance == True:

        print(f"\nIntegrating triples from Flow {batch_offset + 1} to Flow {batch_size+batch_offset}...")
        files, n_triples = integrate_openml_flows(flows_df, flow_params_df, flow_tags_df, flow_dependencies_df,
         targetpath, files, file_part, file_subpart)
        total_triples += n_triples
        print("Integration complete!")
        print("Current dump triple count:", total_triples, "\n")
        file_subpart += 1
        
        if total_triples > goal_triples:
            output_file = "openml_flows_" + str(file_part) + ".nt.gz"
            concatenate_and_compress(targetpath, files, output_file)
            delete_files(targetpath, files)
            file_subpart = 1
            file_part += 1
            total_triples = 0
            files = []

        batch_offset += batch_size

        flows_df, flow_params_df, flow_tags_df, flow_dependencies_df, flow_clearance = get_flow_batch(
        datapath, batch_offset, batch_size)

    if len(files) > 0:
        output_file = "openml_flows_" + str(file_part) + ".nt.gz"
        concatenate_and_compress(targetpath, files, output_file)
        delete_files(targetpath, files)

    print("No more flow data to integrate. Returning...\n")

    return 


def integrate_openml_datasets_from_csv(datapath, targetpath, batch_offset, batch_size, update=False):

    print("Preparing for OpenML Dataset integration...\n")
    (datasets_df, dataset_creators_df, dataset_tags_df,
    dataset_features_df, dataset_references_df, dataset_clearance) = get_dataset_batch(
    datapath, batch_offset, batch_size)
    
    files = []
    if update == False: 
        file_subpart = 1
        file_part = 1
        total_triples = 0
    elif update == True and dataset_clearance == True:
        file_subpart = 2
        file_part = config.OPENML_DATASET_DUMP_PART
        current_dump = "openml_datasets_" + str(file_part) + ".nt"
        total_triples = count_dump_triples(targetpath + current_dump + ".gz")
        unzip_and_save(targetpath + current_dump + ".gz")
        files = [current_dump]   
    
    goal_triples = config.OPENML_DUMP_LIMIT

    while dataset_clearance == True:

        print(f"\nIntegrating triples from Dataset {batch_offset + 1} to Dataset {batch_size+batch_offset}...")
        files, n_triples = integrate_openml_datasets(datasets_df, dataset_creators_df, dataset_tags_df,
        dataset_features_df, dataset_references_df, targetpath, files, file_part, file_subpart)
        total_triples += n_triples
        print("Integration complete!")
        print("Current dump triple count:", total_triples, "\n")
        file_subpart += 1
        
        if total_triples > goal_triples:
            output_file = "openml_datasets_" + str(file_part) + ".nt.gz"
            concatenate_and_compress(targetpath, files, output_file)
            delete_files(targetpath, files)
            file_subpart = 1
            file_part += 1
            total_triples = 0
            files = []

        batch_offset += batch_size

        (datasets_df, dataset_creators_df, dataset_tags_df,
        dataset_features_df, dataset_references_df, dataset_clearance) = get_dataset_batch(
        datapath, batch_offset, batch_size)

    if len(files) > 0:
        output_file = "openml_datasets_" + str(file_part) + ".nt.gz"
        concatenate_and_compress(targetpath, files, output_file)
        delete_files(targetpath, files)

    print("No more dataset data to integrate. Returning...\n")

    return 


def integrate_openml_runs_from_csv(datapath, targetpath, batch_offset, batch_size, update=False):

    print("Preparing for OpenML Run integration...\n")
    run_checkpoint_1, run_checkpoint_2, run_checkpoint_3 = 0, 3162550, 5999999

    full_runs_df = pd.read_csv(datapath + "runs3.csv", dtype={'did': 'Int64',
    'error_message': 'object', 'openml_url': 'object', 'predictions_url': 'object', 'uploader_name': 'object'})
    full_run_evaluations_df = pd.read_csv(datapath + "run_evaluations3.csv")
    full_run_settings_df = pd.read_csv(datapath + "run_settings3.csv")

    runs_df, run_evaluations_df, run_settings_df, run_clearance = get_run_batch(
    datapath, batch_offset, batch_size, full_runs_df, full_run_evaluations_df, full_run_settings_df)

    files = []
    if update == False: 
        file_subpart = 1
        file_part = 1
        total_triples = 0
    elif update == True and run_clearance == True: 
        file_subpart = 2
        file_part = config.OPENML_RUN_DUMP_PART
        current_dump = "openml_runs_" + str(file_part) + ".nt"
        total_triples = count_dump_triples(targetpath + current_dump + ".gz")
        unzip_and_save(targetpath + current_dump + ".gz")
        files = [current_dump]   
    
    goal_triples = config.OPENML_DUMP_LIMIT

    while run_clearance == True:

        print(f"\nIntegrating triples from Run {batch_offset + run_checkpoint_1 + 1} to Run {batch_size + batch_offset + run_checkpoint_1}...")
        files, n_triples = integrate_openml_runs(runs_df, run_evaluations_df, run_settings_df, targetpath, 
        files, file_part, file_subpart)
        total_triples += n_triples
        print("Integration complete!")
        print("Current dump triple count:", total_triples, "\n")
        file_subpart += 1
        
        if total_triples > goal_triples:
            output_file = "openml_runs_" + str(file_part) + ".nt.gz"
            concatenate_and_compress(targetpath, files, output_file)
            delete_files(targetpath, files)
            file_subpart = 1
            file_part += 1
            total_triples = 0
            files = []

        batch_offset += batch_size

        runs_df, run_evaluations_df, run_settings_df, run_clearance = get_run_batch(
        datapath, batch_offset, batch_size, 
        full_runs_df, full_run_evaluations_df, full_run_settings_df)
    
    if len(files) > 0:
        output_file = "openml_runs_" + str(file_part) + ".nt.gz"
        concatenate_and_compress(targetpath, files, output_file)
        delete_files(targetpath, files)

    print("No more run data to integrate. Returning...\n")

    return 


def integrate_openml(update=False):

    datapath = config.OPENML_INPUT
    targetpath = config.OUTPUT_PATH

    if update == False:
        tasks_offset = 0
        flows_offset = 0
        datasets_offset = 0
        runs_offset = 0
    else:
        tasks_offset = config.OPENML_TASK_CHECKPOINT
        flows_offset = config.OPENML_FLOW_CHECKPOINT
        datasets_offset = config.OPENML_DATASET_CHECKPOINT
        runs_offset = config.OPENML_RUN_CHECKPOINT

    batch_size = 1000
    dataset_batch_size = 1 

    integrate_openml_tasks_from_csv(datapath, targetpath, tasks_offset, batch_size, update)
    integrate_openml_flows_from_csv(datapath, targetpath, flows_offset, batch_size, update)
    integrate_openml_datasets_from_csv(datapath, targetpath, datasets_offset, dataset_batch_size, update)
    integrate_openml_runs_from_csv(datapath, targetpath, runs_offset, batch_size, update)

    # Update OpenML integration checkpoints
    runs_csv = datapath + "runs3.csv"
    datasets_csv = datapath + "datasets.csv"
    tasks_csv = datapath + "tasks.csv"
    flows_csv = datapath + "flows.csv"

    checkpoints, latest_ids = get_checkpoints([runs_csv, datasets_csv, tasks_csv, flows_csv])
    run_cp, dataset_cp, task_cp, flow_cp = checkpoints[0], checkpoints[1], checkpoints[2], checkpoints[3]
    config.update_openml_checkpoints(run_cp, dataset_cp, task_cp, flow_cp)

    return

if __name__ == "__main__":

    integrate_openml(update=True)
