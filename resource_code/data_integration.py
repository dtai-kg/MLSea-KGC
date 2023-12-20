import morph_kgc
from get_data_sample import *
from queries import *
import sys
import pandas as pd
import json
import warnings
import gzip
import shutil
import os
from preprocessing_modules import *
warnings.simplefilter(action='ignore', category=FutureWarning)
import config

def integrate_kaggle_kernel(kernels_df, users_df, kernel_versions_df,
 targetpath, files, file_part, file_subpart):
    
    kernels_df['CreationDate'] = kernels_df['CreationDate'].apply(preprocess_datetime)
    kernel_versions_df['CreationDate'] = kernel_versions_df['CreationDate'].apply(preprocess_datetime)
    kernel_versions_df = preprocess_df_strings(kernel_versions_df)
    users_df = preprocess_df_strings(users_df)
    data_dict = {"kernels_df": kernels_df,
                 "users_df": users_df,
                 "kernel_versions_df": kernel_versions_df}
    graph = morph_kgc.materialize('./morph_config/kaggle_kernel_conf.ini', data_dict)
    print("RDF generated!")
    print("Saving to disk...")
    # for s,p,o in graph.triples((None, None, None)):
    #     print(s,p,o)

    filename = "kaggle_kernels_" + str(file_part) + "_part_" + str(file_subpart) + ".nt"
    graph.serialize(destination = targetpath + filename, format = "nt", encoding = "utf-8")
    files.append(filename)

    return files, len(graph)

def integrate_kaggle_dataset(datasets_df, users_df, dataset_versions_df, dataset_tags_df,
 targetpath, files, file_part, file_subpart):
    
    datasets_df['CreationDate'] = datasets_df['CreationDate'].apply(preprocess_datetime)
    dataset_versions_df['CreationDate'] = dataset_versions_df['CreationDate'].apply(preprocess_datetime)
    dataset_versions_df = preprocess_df_strings(dataset_versions_df)
    users_df = preprocess_df_strings(users_df)
    dataset_tags_df = preprocess_df_strings(dataset_tags_df)
    data_dict = {"datasets_df": datasets_df,
                 "users_df": users_df,
                 "dataset_versions_df": dataset_versions_df,
                 "dataset_tags_df": dataset_tags_df}
    graph = morph_kgc.materialize('./morph_config/kaggle_dataset_conf.ini', data_dict)
    print("RDF generated!")
    print("Saving to disk...")
    # for s,p,o in graph.triples((None, None, None)):
    #     print(s,p,o)

    filename = "kaggle_datasets_" + str(file_part) + "_part_" + str(file_subpart) + ".nt"
    graph.serialize(destination = targetpath + filename, format = "nt", encoding = "utf-8")
    files.append(filename)

    return files, len(graph)

def integrate_pwc_object(mapping_config_file, targetpath, 
        files, file_part, file_subpart):

    graph = morph_kgc.materialize(mapping_config_file)
    print("RDF generated!")
    print("Saving to disk...")
    # for s,p,o in graph.triples((None, None, None)):
    #     print(s,p,o)

    filename = "pwc_" + str(file_part) + "_part_" + str(file_subpart) + ".nt"
    graph.serialize(destination = targetpath + filename, format = "nt", encoding = "utf-8")
    files.append(filename)

    return files, len(graph)


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

def integrate_openml_tasks_from_csv(datapath, targetpath, batch_offset, batch_size):

    tasks_df, tasks_clearance = get_task_batch(
    datapath, batch_offset, batch_size)
    file_part = 1
    file_subpart = 1
    total_triples = 0
    goal_triples = 50000000
    files = []

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

def integrate_openml_flows_from_csv(datapath, targetpath, batch_offset, batch_size):

    flows_df, flow_params_df, flow_tags_df, flow_dependencies_df, flow_clearance = get_flow_batch(
    datapath, batch_offset, batch_size)
    file_part = 1
    file_subpart = 1
    total_triples = 0
    goal_triples = 50000000
    files = []

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


def integrate_openml_datasets_from_csv(datapath, targetpath, batch_offset, batch_size):

    (datasets_df, dataset_creators_df, dataset_tags_df,
    dataset_features_df, dataset_references_df, dataset_clearance) = get_dataset_batch(
    datapath, batch_offset, batch_size)
    file_part = 1
    file_subpart = 1
    total_triples = 0
    goal_triples = 25000000
    files = []

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


def integrate_openml_runs_from_csv(datapath, targetpath, batch_offset, batch_size):

    run_checkpoint_1, run_checkpoint_2, run_checkpoint_3 = 0, 3162550, 5999999

    full_runs_df = pd.read_csv(datapath + "runs3.csv", dtype={'did': 'Int64',
    'error_message': 'object', 'openml_url': 'object', 'predictions_url': 'object', 'uploader_name': 'object'})
    full_run_evaluations_df = pd.read_csv(datapath + "run_evaluations3.csv")
    full_run_settings_df = pd.read_csv(datapath + "run_settings3.csv")

    runs_df, run_evaluations_df, run_settings_df, run_clearance = get_run_batch(
    datapath, batch_offset, batch_size, full_runs_df, full_run_evaluations_df, full_run_settings_df)
    file_part = 16
    file_subpart = 1
    total_triples = 0
    goal_triples = 50000000
    files = []

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


def integrate_pwc_from_json_batch(datapath, targetpath, filename, mapping_config_file, batch_size, 
    file_part, file_subpart, total_triples, goal_triples, files):

    batch_offset = 0
    with open(datapath+filename, 'r', encoding='utf-8') as j:
        contents = json.load(j)

    sample_filename = filename.split('.')[0] + "_sample.json"
    batch_clearance = get_pwc_json_batch(sample_filename, contents, batch_offset, batch_size)

    while batch_clearance == True:
        print(f"\nIntegrating triples from PwC {filename} {batch_offset + 1} to PwC {filename} {batch_size + batch_offset}...")
        files, n_triples = integrate_pwc_object(mapping_config_file, targetpath, 
        files, file_part, file_subpart)
        total_triples += n_triples
        print("Integration complete!")
        print("Current dump triple count:", total_triples, "\n")
        file_subpart += 1

        if total_triples > goal_triples:
            output_file = "pwc_" + str(file_part) + ".nt.gz"
            concatenate_and_compress(targetpath, files, output_file)
            delete_files(targetpath, files)
            file_subpart = 1
            file_part += 1
            total_triples = 0
            files = []

        batch_offset += batch_size

        batch_clearance = get_pwc_json_batch(sample_filename, contents, batch_offset, batch_size)

    print("No more data to integrate. Returning...\n")

    return file_part, file_subpart, total_triples, files
    

def integrate_pwc_from_csv(datapath, targetpath, filename, mapping_config_file, batch_size, 
    file_part, file_subpart, total_triples, goal_triples, files):

    df = pd.read_csv(datapath + filename)
    batch_offset = 0
    batch, batch_clearance = get_df_batch(df, batch_offset, batch_size)
    batch.to_csv("Mappings/PwC/Data/evaluations_sample.csv", index=False)

    while batch_clearance == True:
        print(f"\nIntegrating triples from PwC {filename} {batch_offset + batch_size + 1} to PwC {filename} {batch_size + batch_offset}...")
        files, n_triples = integrate_pwc_object(mapping_config_file, targetpath, 
        files, file_part, file_subpart)
        total_triples += n_triples
        print("Integration complete!")
        print("Current dump triple count:", total_triples, "\n")
        file_subpart += 1

        if total_triples > goal_triples:
            output_file = "pwc_" + str(file_part) + ".nt.gz"
            concatenate_and_compress(targetpath, files, output_file)
            delete_files(targetpath, files)
            file_subpart = 1
            file_part += 1
            total_triples = 0
            files = []

        batch_offset += batch_size

        batch, batch_clearance = get_df_batch(df, batch_offset, batch_size)
        batch.to_csv("Mappings/PwC/Data/evaluations_sample.csv", index=False)

    print("No more data to integrate. Returning...\n")

    return file_part, file_subpart, total_triples, files


def integrate_kaggle_datasets_from_csv(datapath, targetpath, offset, batch_size,
    file_part, total_triples, goal_triples, files):

    datasets_df, users_df, dataset_versions_df, dataset_tags_df, tags_df = load_kaggle_dataset_data(datapath)
    
    datasets_df_sample, users_df_sample, dataset_versions_df_sample, dataset_tags_df_sample, dataset_clearance = (
    get_kaggle_dataset_batch(datasets_df, users_df,
    dataset_versions_df, dataset_tags_df, tags_df, offset, batch_size))
    file_subpart = 1

    while dataset_clearance:
        print(f"\nIntegrating triples from Dataset {offset + 1} to Dataset {batch_size + offset}...")
        files, n_triples = integrate_kaggle_dataset(datasets_df_sample, users_df_sample,
        dataset_versions_df_sample, dataset_tags_df_sample, targetpath, files, file_part, file_subpart)
        total_triples += n_triples
        print("Integration complete!")
        print("Current dump triple count:", total_triples, "\n")
        file_subpart += 1

        if total_triples > goal_triples:
            output_file = "kaggle" + str(file_part) + ".nt.gz"
            concatenate_and_compress(targetpath, files, output_file)
            delete_files(targetpath, files)
            file_subpart = 1
            file_part += 1
            total_triples = 0
            files = []

        offset += batch_size

        datasets_df_sample, users_df_sample, dataset_versions_df_sample, dataset_tags_df_sample, dataset_clearance = (
        get_kaggle_dataset_batch(datasets_df, users_df,
        dataset_versions_df, dataset_tags_df, tags_df, offset, batch_size))

    print("No more dataset data to integrate. Exiting...\n")

    return file_part, total_triples, files



def integrate_kaggle_kernels_from_csv(datapath, targetpath, offset, batch_size,
    file_part, total_triples, goal_triples, files):

    (kernels_df, users_df, kernel_versions_df, kernel_version_ds_df,
    dataset_versions_df, kernel_languages_df) = load_kaggle_kernel_data(datapath)
    
    kernels_df_sample, users_df_sample, kernel_versions_df_sample, kernel_clearance = (
        get_kaggle_kernel_batch(kernels_df, users_df, kernel_versions_df, kernel_version_ds_df,
         dataset_versions_df, kernel_languages_df, offset, batch_size) )
    file_subpart = 1

    while kernel_clearance:
        print(f"\nIntegrating triples from Kernel {offset + 1} to Kernel {batch_size + offset}...")
        files, n_triples = integrate_kaggle_kernel(kernels_df_sample, users_df_sample,
        kernel_versions_df_sample, targetpath, files, file_part, file_subpart)
        total_triples += n_triples
        print("Integration complete!")
        print("Current dump triple count:", total_triples, "\n")
        file_subpart += 1

        if total_triples > goal_triples:
            output_file = "kaggle" + str(file_part) + ".nt.gz"
            concatenate_and_compress(targetpath, files, output_file)
            delete_files(targetpath, files)
            file_subpart = 1
            file_part += 1
            total_triples = 0
            files = []

        offset += batch_size

        kernels_df_sample, users_df_sample, kernel_versions_df_sample, kernel_clearance = (
        get_kaggle_kernel_batch(kernels_df, users_df, kernel_versions_df, kernel_version_ds_df,
         dataset_versions_df, kernel_languages_df, offset, batch_size))
        

    print("No more kernel data to integrate. Returning...\n")

    return file_part, total_triples, files

def integrate_pwc():

    print("Processing Papers with Code dumps...")
    datapath = config.PWC_INPUT + config.ORIGINAL_DATA_FOLDER
    filenames = ['datasets.json', 
                'paper_code_links.json', 
                'papers_with_abstracts.json',
                'evaluations.json']
    for file in filenames:
        preprocess_json(datapath, file)
    pre_process_pwc_evaluations(datapath)
    filenames.append('evaluations.csv')
    print("Dumps were succesfully cleaned!\n")

    mappings = ['./morph_config/pwc_dataset_conf.ini',
                './morph_config/pwc_paper_code_links_conf.ini',
                './morph_config/pwc_paper_conf.ini',
                './morph_config/pwc_model_conf.ini',
                './morph_config/pwc_evaluations_conf.ini']
    
    targetpath = config.OUTPUT_PATH
    file_part = 1
    file_subpart = 1
    total_triples = 0
    goal_triples = 50000000
    files = []
    batch_size = 5000
    for i in range(0,len(filenames)):
        
        if filenames[i].split('.')[1] == "json":
            file_part, file_subpart, total_triples, files = integrate_pwc_from_json_batch(
            datapath, targetpath, filenames[i], mappings[i], batch_size, 
            file_part, file_subpart, total_triples, goal_triples, files)
        else:
            file_part, file_subpart, total_triples, files = integrate_pwc_from_csv(
            datapath, targetpath, filenames[i], mappings[i], batch_size, 
            file_part, file_subpart, total_triples, goal_triples, files)
        

    output_file = "pwc_" + str(file_part) + ".nt.gz"
    concatenate_and_compress(targetpath, files, output_file)
    delete_files(targetpath, files)

    return

def integrate_openml():

    datapath = config.OPENML_INPUT + config.ORIGINAL_DATA_FOLDER
    targetpath = config.OUTPUT_PATH
    batch_offset = 0
    batch_size = 1000
    dataset_batch_size = 1
    integrate_openml_tasks_from_csv(datapath, targetpath, batch_offset, batch_size)
    integrate_openml_flows_from_csv(datapath, targetpath, batch_offset, batch_size)
    integrate_openml_datasets_from_csv(datapath, targetpath, batch_offset, batch_size)
    integrate_openml_runs_from_csv(datapath, targetpath, batch_offset, dataset_batch_size)

    return

def integrate_kaggle():

    datapath = config.KAGGLE_INPUT + config.ORIGINAL_DATA_FOLDER
    targetpath = config.OUTPUT_PATH
    file_part = 1
    total_triples = 0
    goal_triples = 50000000
    files = []
    
    offset = 0
    size = 10000
    file_part, total_triples, files = integrate_kaggle_datasets_from_csv(
    datapath, targetpath, offset, size,
    file_part, total_triples, goal_triples, files)

    offset = 0
    size = 10000
    file_part, total_triples, files = integrate_kaggle_kernels_from_csv(
    datapath, targetpath, offset, size, 
    file_part, total_triples, goal_triples, files)

    output_file = "kaggle_" + str(file_part) + ".nt.gz"
    concatenate_and_compress(targetpath, files, output_file)
    delete_files(targetpath, files)

    return


def concatenate_and_compress(targetpath, files, output_file):
    print("\nConcatenating and compressing dumps...")
    with open(targetpath + output_file, 'wb') as out_file, \
         gzip.open(out_file, 'wt', encoding='utf-8') as zip_file:
        for file_name in files:
            with open(targetpath + file_name, 'rt', encoding='utf-8') as in_file:
                shutil.copyfileobj(in_file, zip_file)

def delete_files(targetpath, files):
    for file_name in files:
        os.remove(targetpath + file_name)

if __name__ == "__main__":

    integrate_openml()
    integrate_kaggle()
    integrate_pwc()








    

    