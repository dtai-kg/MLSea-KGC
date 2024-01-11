import morph_kgc
from get_data_sample import load_kaggle_dataset_data, load_kaggle_kernel_data, get_kaggle_dataset_batch, get_kaggle_kernel_batch
from queries import *
import pandas as pd
import warnings
from dump_storage import *
from preprocessing_modules import *
warnings.simplefilter(action='ignore', category=FutureWarning)

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

def integrate_kaggle_datasets_from_csv(datapath, targetpath, offset, batch_size,
    file_part, total_triples, goal_triples, files, update=False):

    datasets_df, users_df, dataset_versions_df, dataset_tags_df, tags_df = load_kaggle_dataset_data(datapath, update)

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
    file_part, total_triples, goal_triples, files, update=False):

    (kernels_df, users_df, kernel_versions_df, kernel_version_ds_df,
    dataset_versions_df, kernel_languages_df) = load_kaggle_kernel_data(datapath, update)
    
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


def integrate_kaggle(update = False):

    print("Preparing Kaggle integration...")
    datapath = config.KAGGLE_INPUT
    targetpath = config.OUTPUT_PATH

    if update == False:
        file_part = 1
        total_triples = 0
        files = []
    elif update == True: 
        file_part = config.KAGGLE_DUMP_PART
        current_dump = "kaggle_" + str(file_part) + ".nt"
        total_triples = count_dump_triples(targetpath + current_dump + ".gz")
        unzip_and_save(targetpath + current_dump + ".gz")
        files = [current_dump]  
    
    goal_triples = config.KAGGLE_DUMP_LIMIT
    
    print("Initializing Kaggle Dataset integration...\n")
    offset = 0
    size = 10000
    file_part, total_triples, files = integrate_kaggle_datasets_from_csv(
    datapath, targetpath, offset, size,
    file_part, total_triples, goal_triples, files, update)

    print("Initializing Kaggle Kernel integration...\n")
    offset = 0
    size = 10000
    file_part, total_triples, files = integrate_kaggle_kernels_from_csv(
    datapath, targetpath, offset, size, 
    file_part, total_triples, goal_triples, files, update)

    if len(files)>0:
        output_file = "kaggle_" + str(file_part) + ".nt.gz"
        concatenate_and_compress(targetpath, files, output_file)
        delete_files(targetpath, files)

    return


if __name__ == "__main__":

    integrate_kaggle(update = True)