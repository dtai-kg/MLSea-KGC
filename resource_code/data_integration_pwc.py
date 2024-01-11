import morph_kgc
from get_data_sample import get_df_batch, get_pwc_json_batch
from queries import *
import sys
import pandas as pd
import json
import warnings
import re
from dump_storage import *
from preprocessing_modules import *
from update_sources import *
warnings.simplefilter(action='ignore', category=FutureWarning)

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


def integrate_pwc(update = False):

    print("Processing Papers with Code dumps...")

    targetpath = config.OUTPUT_PATH
    goal_triples = config.PWC_DUMP_LIMIT
    original_path = config.PWC_INPUT + config.ORIGINAL_DATA_FOLDER
    update_path = config.PWC_INPUT + config.UPDATE_MONTH_FOLDER
    updates_folder = "Updates/"

    if update == False:
        file_part = 1
        file_subpart = 1
        total_triples = 0
        files = []
        datapath = original_path
    else:
        datapath = update_path + updates_folder
        file_subpart = 2
        file_part = config.PWC_DUMP_PART
        current_dump = "pwc_" + str(file_part) + ".nt"
        total_triples = count_dump_triples(targetpath + current_dump + ".gz")
        unzip_and_save(targetpath + current_dump + ".gz")
        files = [current_dump]   
    

    filenames = ['datasets.json', 
                'paper_code_links.json', 
                'papers_with_abstracts.json',
                'evaluations.json']
    for file in filenames:
        preprocess_json(datapath, file)
        if update == True: 
            get_json_updates(original_path + file, update_path + file,
            update_path + updates_folder + file)

    pre_process_pwc_evaluations(datapath)
    filenames.append('evaluations.csv')
    print("Dumps were succesfully cleaned!\n")

    mappings = ['./morph_config/pwc_dataset_conf.ini',
                './morph_config/pwc_paper_code_links_conf.ini',
                './morph_config/pwc_paper_conf.ini',
                './morph_config/pwc_model_conf.ini',
                './morph_config/pwc_evaluations_conf.ini']
    
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
        

    if len(files) > 0: 
        output_file = "pwc_" + str(file_part) + ".nt.gz"
        concatenate_and_compress(targetpath, files, output_file)
        delete_files(targetpath, files)

    return


if __name__ == "__main__":

    integrate_pwc(update = True)