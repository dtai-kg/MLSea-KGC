import morph_kgc
from get_data_sample import get_df_batch, get_task_batch, get_dataset_batch, get_run_batch, get_flow_batch, get_pwc_json_batch
from get_data_sample import load_kaggle_dataset_data, load_kaggle_kernel_data, get_kaggle_dataset_batch, get_kaggle_kernel_batch
from queries import *
from graph_db import *
import sys
import pandas as pd
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def integrate_kaggle_kernel(kernels_df, users_df, kernel_versions_df, db, named_graph):
    host = GRAPH_DB_HOST
    repository = KAGGLE_REPOSITORY
    db = GraphDB_SW(host, repository)
    
    data_dict = {"kernels_df": kernels_df,
                 "users_df": users_df,
                 "kernel_versions_df": kernel_versions_df}
    graph = morph_kgc.materialize('./morph_config/kaggle_kernel_conf.ini', data_dict)
    # for s,p,o in graph.triples((None, None, None)):
    #     print(s,p,o)
    print("RDF generated! Uploading to database...")
    response = db.insert_data(named_graph, graph)

    return response

def integrate_kaggle_dataset(datasets_df, users_df, dataset_versions_df, dataset_tags_df, db, named_graph):
    
    data_dict = {"datasets_df": datasets_df,
                 "users_df": users_df,
                 "dataset_versions_df": dataset_versions_df,
                 "dataset_tags_df": dataset_tags_df}
    graph = morph_kgc.materialize('./morph_config/kaggle_dataset_conf.ini', data_dict)
    # for s,p,o in graph.triples((None, None, None)):
    #     print(s,p,o)
    print("RDF generated! Uploading to database...")
    response = db.insert_data(named_graph, graph)

    return response

def integrate_pwc_object(mapping_config_file, db, named_graph):

    graph = morph_kgc.materialize(mapping_config_file)
    print("RDF generated! Uploading to database...")
    # for s,p,o in graph.triples((None, None, None)):
    #     print(s,p,o)
    db.insert_data(named_graph, graph)

    return


def integrate_openml_tasks(tasks_df, db, named_graph):
    
    data_dict = {"tasks_df": tasks_df}
    graph = morph_kgc.materialize('./morph_config/task_conf.ini', data_dict)
    print("RDF generated! Uploading to database...")
    db.insert_data(named_graph, graph)

    return 


def integrate_openml_datasets(
    datasets_df, dataset_creators_df, dataset_tags_df,
    dataset_features_df, dataset_references_df, db, named_graph):

    data_dict = {"datasets_df": datasets_df,
                "dataset_creators_df": dataset_creators_df,
                "dataset_tags_df": dataset_tags_df,
                "dataset_features_df": dataset_features_df,
                "dataset_references_df": dataset_references_df}
    graph = morph_kgc.materialize('./morph_config/dataset_conf.ini', data_dict)
    print("RDF generated! Uploading to database...")
    db.insert_data(named_graph, graph)

    return 


def integrate_openml_runs(
    runs_df, run_evaluations_df, run_settings_df, db, named_graph):

    runs_df['upload_time'] = runs_df['upload_time'].apply(preprocess_datetime)

    data_dict = {"runs_df": runs_df,
                "run_evaluations_df": run_evaluations_df,
                "run_settings_df": run_settings_df}
    graph = morph_kgc.materialize('./morph_config/run_conf.ini', data_dict)
    print("RDF generated! Uploading to database...")
    db.insert_data(named_graph, graph)

    return 


def integrate_openml_flows(
    flows_df, flow_params_df, flow_tags_df, flow_dependencies_df, db, named_graph):

    data_dict = {"flows_df": flows_df,
                "flow_params_df": flow_params_df,
                "flow_tags_df": flow_tags_df,
                "flow_dependencies_df": flow_dependencies_df}
    graph = morph_kgc.materialize('./morph_config/flow_conf.ini', data_dict)
    print("RDF generated! Uploading to database...")
    db.insert_data(named_graph, graph)

    return 


# def integrate_openml(datapath, 
#     run_batch_offset, run_batch_size,
#     dataset_batch_offset, dataset_batch_size,
#     task_batch_offset, task_batch_size,
#     flow_batch_offset, flow_batch_size):

#     # tasks_df, tasks_clearance = get_task_batch(
#     # datapath, task_batch_offset, task_batch_size)
#     # (datasets_df, dataset_creators_df, dataset_tags_df,
#     # dataset_features_df, dataset_references_df, dataset_clearance) = get_dataset_batch(
#     # datapath, dataset_batch_offset, dataset_batch_size)
#     # runs_df, run_evaluations_df, run_settings_df, run_clearance = get_run_batch(
#     # datapath, run_batch_offset, run_batch_size)
#     flows_df, flow_params_df, flow_tags_df, flow_dependencies_df, flow_clearance = get_flow_batch(
#     datapath, flow_batch_offset, flow_batch_size)

#     # task_graph = integrate_openml_tasks(tasks_df, tasks_clearance)

#     # dataset_graph = integrate_openml_datasets (
#     # datasets_df, dataset_creators_df, dataset_tags_df,
#     # dataset_features_df, dataset_references_df, dataset_clearance)

#     # run_graph = integrate_openml_runs(
#     # runs_df, run_evaluations_df, run_settings_df, run_clearance)

#     flow_graph = integrate_openml_flows(
#     flows_df, flow_params_df, flow_tags_df, flow_dependencies_df, flow_clearance)

#     return flow_graph

def find_instance_count(db, graph):

    query = get_query(SELECT_ID_COUNT, graph)
    try:
        count = int(db.get_triples(query)["results"]["bindings"][0]["instanceCount"]["value"])
    except:
        print("Connection problem! Returning...")
        sys.exit()

    return count

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

def integrate_openml_tasks_from_csv(datapath, db, batch_size):

    named_graph = OPENML_TASK_GRAPH

    task_batch_size, task_batch_offset = batch_size, find_instance_count(db, named_graph)
    tasks_df, tasks_clearance = get_task_batch(
    datapath, task_batch_offset, task_batch_size)

    while tasks_clearance == True:
        
        print(f"\nIntegrating triples from Task {task_batch_offset + 1} to Task {task_batch_size+task_batch_offset}...")
        integrate_openml_tasks(tasks_df, db, named_graph)
        print("Integration complete!\n")

        task_batch_offset += task_batch_size

        tasks_df, tasks_clearance = get_task_batch(
        datapath, task_batch_offset, task_batch_size)

    print("No more task data to integrate. Returning...\n")

    return 

def integrate_openml_flows_from_csv(datapath, db, batch_size):

    named_graph = OPENML_FLOW_GRAPH

    flow_batch_size, flow_batch_offset = batch_size, find_instance_count(db, named_graph)
    flows_df, flow_params_df, flow_tags_df, flow_dependencies_df, flow_clearance = get_flow_batch(
    datapath, flow_batch_offset, flow_batch_size)

    while flow_clearance == True:

        print(f"\nIntegrating triples from Flow {flow_batch_offset + 1} to Flow {flow_batch_size+flow_batch_offset}...")
        integrate_openml_flows(flows_df, flow_params_df, flow_tags_df, flow_dependencies_df, db, named_graph)
        print("Integration complete!\n")

        flow_batch_offset += flow_batch_size

        flows_df, flow_params_df, flow_tags_df, flow_dependencies_df, flow_clearance = get_flow_batch(
        datapath, flow_batch_offset, flow_batch_size)

    print("No more flow data to integrate. Returning...\n")

    return 

def integrate_openml_datasets_from_csv(datapath, db, batch_size):

    named_graph = OPENML_DATASET_GRAPH

    dataset_batch_size, dataset_batch_offset = batch_size, find_instance_count(db, named_graph)
    (datasets_df, dataset_creators_df, dataset_tags_df,
    dataset_features_df, dataset_references_df, dataset_clearance) = get_dataset_batch(
    datapath, dataset_batch_offset, dataset_batch_size)

    while dataset_clearance == True:

        print(f"\nIntegrating triples from Dataset {dataset_batch_offset + 1} to Dataset {dataset_batch_size+dataset_batch_offset}...")
        integrate_openml_datasets(datasets_df, dataset_creators_df, dataset_tags_df,
        dataset_features_df, dataset_references_df, db, named_graph)
        print("Integration complete!\n")

        dataset_batch_offset += dataset_batch_size

        (datasets_df, dataset_creators_df, dataset_tags_df,
        dataset_features_df, dataset_references_df, dataset_clearance) = get_dataset_batch(
        datapath, dataset_batch_offset, dataset_batch_size)

    print("No more flow data to integrate. Returning...\n")

    return 


def integrate_openml_runs_from_csv(datapath, db, batch_size):

    named_graph = OPENML_RUN_GRAPH
    run_checkpoint_1, run_checkpoint_2 = 3162550, 5999999

    full_runs_df = pd.read_csv(datapath + "runs3.csv", dtype={'did': 'Int64',
    'error_message': 'object', 'openml_url': 'object', 'predictions_url': 'object', 'uploader_name': 'object'})
    full_run_evaluations_df = pd.read_csv(datapath + "run_evaluations3.csv")
    full_run_settings_df = pd.read_csv(datapath + "run_settings3.csv")

    run_batch_size, run_batch_offset = batch_size, find_instance_count(db, named_graph)
    run_batch_offset = run_batch_offset - run_checkpoint_2
    runs_df, run_evaluations_df, run_settings_df, run_clearance = get_run_batch(
    datapath, run_batch_offset, run_batch_size, full_runs_df, full_run_evaluations_df, full_run_settings_df)

    while run_clearance == True:

        print(f"\nIntegrating triples from Run {run_batch_offset + run_checkpoint_2 + 1} to Run {run_batch_size + run_batch_offset + run_checkpoint_2}...")
        integrate_openml_runs(runs_df, run_evaluations_df, run_settings_df, db, named_graph)
        print("Integration complete!\n")

        run_batch_offset += run_batch_size

        runs_df, run_evaluations_df, run_settings_df, run_clearance = get_run_batch(
        datapath, run_batch_offset, run_batch_size, full_runs_df, full_run_evaluations_df, full_run_settings_df)

    print("No more run data to integrate. Returning...\n")

    return 


def integrate_pwc_from_json_batch(datapath, filename, db, mapping_config_file, batch_size):

    named_graph = PWC_GRAPH
    batch_offset = 0
    with open(datapath+filename, 'r', encoding='utf-8') as j:
        contents = json.load(j)

    sample_filename = filename.split('.')[0] + "_sample.json"
    batch_clearance = get_pwc_json_batch(sample_filename, contents, batch_offset, batch_size)

    while batch_clearance == True:
        print(f"\nIntegrating triples from PwC {filename} {batch_offset + 1} to PwC {filename} {batch_size + batch_offset}...")
        integrate_pwc_object(mapping_config_file, db, named_graph)
        print("Integration complete!\n")

        batch_offset += batch_size

        batch_clearance = get_pwc_json_batch(sample_filename, contents, batch_offset, batch_size)

    print("No more data to integrate. Returning...\n")

    return

def integrate_pwc_from_csv(datapath, filename, db, mapping_config_file, batch_size):

    named_graph = PWC_GRAPH
    batch_offset = 0
    df = pd.read_csv(datapath + filename)
    batch, batch_clearance = get_df_batch(df, batch_offset, batch_size)
    batch.to_csv("Mappings/PwC/Data/evaluations_sample.csv", index=False)

    while batch_clearance == True:
        print(f"\nIntegrating triples from PwC {filename} {batch_offset + batch_size + 1} to PwC {filename} {batch_size + batch_offset}...")
        integrate_pwc_object(mapping_config_file, db, named_graph)
        print("Integration complete!\n")

        batch_offset += batch_size

        batch, batch_clearance = get_df_batch(df, batch_offset, batch_size)

    print("No more data to integrate. Returning...\n")

    return

def integrate_kaggle_datasets_from_csv(datapath, db, offset, batch_size):

    named_graph = KAGGLE_GRAPH
    datasets_df, users_df, dataset_versions_df, dataset_tags_df, tags_df = load_kaggle_dataset_data(datapath)
    
    datasets_df_sample, users_df_sample, dataset_versions_df_sample, dataset_tags_df_sample, dataset_clearance = (
    get_kaggle_dataset_batch(datasets_df, users_df,
    dataset_versions_df, dataset_tags_df, tags_df, offset, batch_size))

    while dataset_clearance:
        print(f"\nIntegrating triples from Dataset {offset + 1} to Dataset {batch_size + offset}...")
        response = integrate_kaggle_dataset(datasets_df_sample, users_df_sample, dataset_versions_df_sample, dataset_tags_df_sample, db, named_graph)
        
        if response == 200 or response == 204:
            print("Integration complete!\n")

            offset += batch_size

            datasets_df_sample, users_df_sample, dataset_versions_df_sample, dataset_tags_df_sample, dataset_clearance = (
            get_kaggle_dataset_batch(datasets_df, users_df,
            dataset_versions_df, dataset_tags_df, tags_df, offset, batch_size))
        
        else:
            print("Integration failed! Exiting... \n")
            #integrate_kaggle(offset)
            return

    print("No more dataset data to integrate. Returning...\n")

    return

def integrate_kaggle_kernels_from_csv(datapath, db, offset, batch_size):

    named_graph = KAGGLE_GRAPH
    kernels_df, users_df, kernel_versions_df, kernel_version_ds_df, dataset_versions_df, kernel_languages_df = load_kaggle_kernel_data(datapath)
    
    kernels_df_sample, users_df_sample, kernel_versions_df_sample, kernel_clearance = (
        get_kaggle_kernel_batch(kernels_df, users_df, kernel_versions_df, kernel_version_ds_df, dataset_versions_df, kernel_languages_df,
                                offset, batch_size)
    )
    

    while kernel_clearance:
        print(f"\nIntegrating triples from Kernel {offset + 1} to Kernel {batch_size + offset}...")
        response = integrate_kaggle_kernel(kernels_df_sample, users_df_sample, kernel_versions_df_sample, db, named_graph)
        
        if response == 200 or response == 204:
            print("Integration complete!\n")

            offset += batch_size

            kernels_df_sample, users_df_sample, kernel_versions_df_sample, kernel_clearance = (
            get_kaggle_kernel_batch(kernels_df, users_df, kernel_versions_df, kernel_version_ds_df, dataset_versions_df, kernel_languages_df,
                                    offset, batch_size)
            )

        else:
            print("Integration failed! Exiting... \n")
            #integrate_kaggle(offset)
            return
        

    print("No more kernel data to integrate. Returning...\n")

    return

def integrate_pwc():

    datapath = "../Data/PwC-Data/"
    host = GRAPH_DB_HOST
    repository = PWC_REPOSITORY
    db = GraphDB_SW(host, repository)
    batch_size = 50

    filenames = ['datasets.json', 
                'paper_code_links.json', 
                'papers_with_abstracts.json',
                'evaluations.json', 
                'evaluations.csv']
    mappings = ['./morph_config/pwc_dataset_conf.ini',
                './morph_config/pwc_paper_code_links_conf.ini',
                './morph_config/pwc_paper_conf.ini',
                './morph_config/pwc_model_conf.ini',
                './morph_config/pwc_evaluations_conf.ini']
    
    for i in range(2,len(filenames)):
        
        if filenames[i].split('.')[1] == "json":
            integrate_pwc_from_json_batch(datapath, filenames[i], db, mappings[i], batch_size)
        else:
            integrate_pwc_from_csv(datapath, filenames[i], db, mappings[i], batch_size)

    return

def integrate_openml():

    # integrate_openml_tasks_from_csv(datapath, db, batch_size)
    # integrate_openml_flows_from_csv(datapath, db, batch_size)
    # integrate_openml_datasets_from_csv(datapath, db, batch_size)
    # integrate_openml_runs_from_csv(datapath, db, batch_size)

    return

def integrate_kaggle(offset):

    datapath = "../Data/Kaggle-Data/"
    #samples_path = "Mappings/Kaggle/Data/"
    host = GRAPH_DB_HOST
    repository = KAGGLE_REPOSITORY
    named_graph = KAGGLE_GRAPH
    db = GraphDB_SW(host, repository)
    
    # size = 100
    # integrate_kaggle_datasets_from_csv(datapath, db, offset, size)

    size = 1000
    integrate_kaggle_kernels_from_csv(datapath, db, offset, size)

    return

def preprocess_datetime(input_string):

    step1 = input_string.replace(' ', 'T')

    return step1



# if __name__ == "__main__":

#     integrate_openml()
    

    