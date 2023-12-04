import os
import pandas as pd
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from datetime import datetime
from preprocessing_modules import *

def get_openml_random_sample(initial_sample_size, random_sample_size):

    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_path = current_dir + "/OpenML-Data/"
    samples_path = data_path + "Samples/"
    files = os.listdir(data_path)
 
    for filename in files:
        if filename[-4:] == ".csv":
            if filename.split("/")[-1] == "runs.csv":
                df = pd.read_csv(data_path + filename, dtype={'did': 'Int64'})
            else:
                df = pd.read_csv(data_path + filename)
            df1 = df.iloc[:initial_sample_size].copy()
            df2 = df.sample(frac=1).iloc[:random_sample_size].copy()
            sample_df = pd.concat([df1, df2], ignore_index=True, axis=0)
            print(sample_df)
            sample_df.to_csv(samples_path + filename, index=False)

    print("Samples generated!")
    return

def get_openml_batches(filepath,
    run_batch_offset, run_batch_size,
    dataset_batch_offset, dataset_batch_size,
    task_batch_offset, task_batch_size,
    flow_batch_offset, flow_batch_size):

    #tasks_df, tasks_clearance = get_tasks_batch(data_path, task_batch_offset, task_batch_size)
    # (datasets_df, dataset_creators, dataset_tags,
    # dataset_features, dataset_clearance, dataset_references) = get_dataset_batch(
    # filepath, dataset_batch_offset, dataset_batch_size)
    # (runs_df, run_evaluations_df, run_settings_df, run_clearance) = get_run_batch(
    # filepath, run_batch_offset, run_batch_size)
    (flows_df, flow_params_df, flow_tags_df, flow_dependencies_df, flow_clearance) = get_flow_batch(
    filepath, flow_batch_offset, flow_batch_size)
    
    return


def get_task_batch(data_path, offset, size):

        tasks_df = pd.read_csv(data_path + "tasks.csv")
        tasks_df, task_clearance = get_df_batch(tasks_df, offset, size)
        
        return tasks_df, task_clearance


def get_dataset_batch(data_path, offset, size):

    datasets_df = pd.read_csv(data_path + "datasets.csv")
    datasets_df, dataset_clearance = get_df_batch(datasets_df, offset, size)
    
    dataset_creators = pd.read_csv(data_path + "dataset_creators.csv")
    dataset_creators = dataset_creators[dataset_creators["did_per_creator"].isin(datasets_df["did"])]

    dataset_tags = pd.read_csv(data_path + "dataset_tags.csv")
    dataset_tags = dataset_tags[dataset_tags["did_per_tag"].isin(datasets_df["did"])]

    dataset_features = pd.read_csv(data_path + "dataset_features.csv")
    dataset_features = dataset_features[dataset_features["dataset_id"].isin(datasets_df["did"])]

    dataset_references = pd.read_csv(data_path + "dataset_references.csv")
    dataset_references = dataset_references[dataset_references["did"].isin(datasets_df["did"])]

    return datasets_df, dataset_creators, dataset_tags, dataset_features, dataset_references, dataset_clearance


def get_run_batch(data_path, offset, size, full_runs_df=None, full_run_evals_df=None, full_run_sets_df = None):

    # runs_df = pd.read_csv(data_path + "runs.csv", dtype={'did': 'Int64',
    # 'error_message': 'object', 'openml_url': 'object', 'predictions_url': 'object', 'uploader_name': 'object'})
    runs_df, run_clearance = get_df_batch(full_runs_df, offset, size)

    # run_evaluations_df = pd.read_csv(data_path + "run_evaluations.csv")
    run_evaluations_df = full_run_evals_df[full_run_evals_df["run_id"].isin(runs_df["run_id"])].copy()

    # run_settings_df = pd.read_csv(data_path + "run_settings.csv")
    run_settings_df = full_run_sets_df[full_run_sets_df["run_id"].isin(runs_df["run_id"])].copy()

    return runs_df, run_evaluations_df, run_settings_df, run_clearance


def get_flow_batch(data_path, offset, size):

    flows_df = pd.read_csv(data_path + "flows.csv")
    flows_df, flow_clearance = get_df_batch(flows_df, offset, size)

    flow_params_df = pd.read_csv(data_path + "flow_params.csv")
    flow_params_df = flow_params_df[flow_params_df["flow_id"].isin(flows_df["id"])]

    flow_tags_df = pd.read_csv(data_path + "flow_tags.csv")
    flow_tags_df = flow_tags_df[flow_tags_df["flow_id"].isin(flows_df["id"])]

    flow_dependencies_df = pd.read_csv(data_path + "flow_dependencies.csv")
    flow_dependencies_df = flow_dependencies_df[flow_dependencies_df["flow_id"].isin(flows_df["id"])]

    return flows_df, flow_params_df, flow_tags_df, flow_dependencies_df, flow_clearance

def get_df_batch(df, offset, size):

    integration_clearance = True
    df_size = len(df)

    if (offset+size) > df_size and offset < df_size:
        df = df.iloc[offset:].copy()
    elif offset > df_size: 
        integration_clearance = False
    elif offset+size < df_size:
        df = df.iloc[offset:(offset+size)].copy()

    return df, integration_clearance

# def save_pwc_sample(datapath):

#     json_file_path = "evaluations.json"
#     sample_file_path = "evaluation_sample.json"

#     with open(datapath+json_file_path, 'r') as j:
#         contents = json.loads(j.read())
#     with open(datapath+sample_file_path, 'w') as f:
#         json.dump(contents[0:2], f, ensure_ascii=False, indent=4)

#     return


def get_pwc_json_batch(sample_filename, contents, offset, size):

    integration_clearance = True
    json_size = len(contents)
    storage_dir = "Mappings/PwC/Data/"

    if (offset+size) > json_size and offset < json_size:
        batch = contents[offset:]
    elif offset > json_size: 
        batch = contents
        integration_clearance = False
    elif offset+size < json_size:
        batch = contents[offset:(offset+size)]

    with open(storage_dir+sample_filename, 'w') as f:
        json.dump(batch, f, ensure_ascii=False, indent=4)

    return integration_clearance

def get_kaggle_dataset_batch(datasets_df, users_df, dataset_versions_df, dataset_tags_df, tags_df, 
                             offset, size):

    datasets_df, dataset_clearance = get_df_batch(datasets_df, offset, size)
    
    users_df = users_df[users_df["Id"].isin(datasets_df["CreatorUserId"])]
    users_df2 = users_df.copy()
    users_df2.rename(columns={'Id': 'CreatorUserId'}, inplace=True)
    
    dataset_versions_df = dataset_versions_df[dataset_versions_df["DatasetId"].isin(datasets_df["Id"])]
    #Keeping only the most recent dataset versions
    pd.to_datetime(dataset_versions_df['CreationDate'])
    indices = dataset_versions_df.groupby('DatasetId')['CreationDate'].idxmax()
    dataset_versions_df = dataset_versions_df.loc[indices]
    dataset_versions_df2 = dataset_versions_df.copy()[["DatasetId", "Slug"]]
    dataset_versions_df2.rename(columns={'DatasetId': 'Id'}, inplace=True)

    # Merging datasets with users and slugs
    datasets_df = pd.merge(datasets_df, users_df2, on='CreatorUserId', how='left')
    datasets_df = datasets_df.drop(["DisplayName"], axis=1)
    datasets_df = pd.merge(datasets_df, dataset_versions_df2, on='Id', how='left')

    dataset_tags_df = dataset_tags_df[dataset_tags_df["DatasetId"].isin(datasets_df["Id"])]

    tags_df_full = pd.concat([tags_df[tags_df["Id"].isin(dataset_tags_df["TagId"])],
    tags_df[tags_df["ParentTagId"].isin(dataset_tags_df["TagId"])]])
    tags_df_full = tags_df_full.drop_duplicates()
    tags_df_full_2 = tags_df_full.copy()[["Id", "Name"]]
    tags_df_full_2.rename(columns={'Id': 'TagId', 'Name': 'TagName'}, inplace=True)

    # Merging datasets with tags
    dataset_tags_df = pd.merge(dataset_tags_df, tags_df_full_2, on='TagId', how='left')

    datasets_df['CreationDate'] = datasets_df['CreationDate'].apply(lambda x: datetime.strptime(str(x), "%m/%d/%Y %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%S") if pd.notna(x) else x)
    dataset_versions_df['CreationDate'] = dataset_versions_df['CreationDate'].apply(lambda x: datetime.strptime(str(x), "%m/%d/%Y %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%S") if pd.notna(x) else x)
    
    return datasets_df, users_df, dataset_versions_df, dataset_tags_df, dataset_clearance

def get_kaggle_kernel_batch(kernels_df, users_df, kernel_versions_df, kernel_version_ds_df, dataset_versions_df, kernel_languages_df,
                            offset, size):

    kernels_df, kernel_clearance = get_df_batch(kernels_df, offset, size)

    users_df = users_df[users_df["Id"].isin(kernels_df["AuthorUserId"])]
    users_df2 = users_df.copy()
    users_df2.rename(columns={'Id': 'AuthorUserId'}, inplace=True)

    # Merging dataset with usernames
    kernels_df = pd.merge(kernels_df, users_df2, on='AuthorUserId', how='left')
    kernels_df = kernels_df.drop(["DisplayName"], axis=1)
    kernels_df['CreationDate'] = kernels_df['CreationDate'].apply(lambda x: datetime.strptime(str(x), "%m/%d/%Y %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%S") if pd.notna(x) else x) 
    del users_df2

    kernel_versions_df = kernel_versions_df[kernel_versions_df["ScriptId"].isin(kernels_df["Id"])]
    # Keeping only the most recent kernel versions
    pd.to_datetime(kernel_versions_df['CreationDate'])
    indices = kernel_versions_df.groupby('ScriptId')['CreationDate'].idxmax()
    kernel_versions_df = kernel_versions_df.loc[indices].copy()
    # Merging kernel versions with script languages
    kernel_languages_df.rename(columns={'Id': 'ScriptLanguageId', 'DisplayName': 'LanguageName'}, inplace=True)
    kernel_versions_df = pd.merge(kernel_versions_df, kernel_languages_df, on='ScriptLanguageId', how='left')
    kernel_versions_df['CreationDate'] = kernel_versions_df['CreationDate'].apply(lambda x: datetime.strptime(str(x), "%m/%d/%Y %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%S") if pd.notna(x) else x)

    kernel_version_ds_df = kernel_version_ds_df[kernel_version_ds_df["KernelVersionId"].isin(kernel_versions_df["Id"])]

    dataset_versions_df = dataset_versions_df[dataset_versions_df["Id"].isin(kernel_version_ds_df["SourceDatasetVersionId"])]
    #Keeping only the most recent dataset versions
    pd.to_datetime(dataset_versions_df['CreationDate'])
    indices = dataset_versions_df.groupby('DatasetId')['CreationDate'].idxmax()
    dataset_versions_df = dataset_versions_df.loc[indices].copy()
    dataset_versions_df['CreationDate'] = dataset_versions_df['CreationDate'].apply(lambda x: datetime.strptime(str(x), "%m/%d/%Y %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%S") if pd.notna(x) else x)
    
    # Merge kernels with related datasets
    dataset_versions_df2 = dataset_versions_df.copy()[["Id", "DatasetId"]]
    dataset_versions_df2.rename(columns={'Id': 'SourceDatasetVersionId'}, inplace=True)
    kernel_version_ds_df = pd.merge(kernel_version_ds_df, dataset_versions_df2, on='SourceDatasetVersionId', how='left')
    del dataset_versions_df2
    kernel_version_ds_df = kernel_version_ds_df[["KernelVersionId", "DatasetId"]]
    kernel_version_ds_df.rename(columns={'KernelVersionId': 'Id'}, inplace=True)
    kernel_version_ds_df['DatasetId'] = kernel_version_ds_df['DatasetId'].apply(lambda x: str(x).split('.')[0] if pd.notna(x) else x)
    kernel_versions_df = pd.merge(kernel_versions_df, kernel_version_ds_df, on='Id', how='left')

    return kernels_df, users_df, kernel_versions_df, kernel_clearance

def load_kaggle_dataset_data(datapath):

    datasets_df = pd.read_csv(datapath + "Datasets.csv")[[
    "Id", "CreatorUserId", "CurrentDatasetVersionId",
     "CreationDate", "TotalViews", "TotalDownloads", "TotalKernels"]]
    
    users_df = pd.read_csv(datapath + "Users.csv")[["Id", "UserName", "DisplayName"]]
    
    dataset_versions_df = pd.read_csv(datapath + "DatasetVersions.csv")[[
    "Id", "DatasetId", "DatasourceVersionId", "CreatorUserId",
    "LicenseName", "CreationDate", "Title", "Slug", "Description"]]
    
    dataset_tags_df = pd.read_csv(datapath + "DatasetTags.csv")
    
    tags_df = pd.read_csv(datapath + "Tags.csv")[[
    "Id", "ParentTagId", "Name"]]

    return datasets_df, users_df, dataset_versions_df, dataset_tags_df, tags_df

def load_kaggle_kernel_data(datapath):

    users_df = pd.read_csv(datapath + "Users.csv")[["Id", "UserName", "DisplayName"]]

    kernels_df = pd.read_csv(datapath + "Kernels.csv")[[
    "Id", "AuthorUserId", "CurrentKernelVersionId",
    "CreationDate", "CurrentUrlSlug"
    ]]
    #Filtering of kernels from restricted accounts
    kernels_df = kernels_df[kernels_df["AuthorUserId"].isin(users_df["Id"])]

    kernel_versions_df = pd.read_csv(datapath + "KernelVersions.csv")[[
    "Id", "ScriptId", "ScriptLanguageId", "AuthorUserId", 
    "CreationDate", "Title"
    ]]
    #Filtering of kernels from restricted accounts
    kernel_versions_df = kernel_versions_df[kernel_versions_df["AuthorUserId"].isin(users_df["Id"])]

    kernel_version_ds_df = pd.read_csv(datapath + "KernelVersionDatasetSources.csv")

    dataset_versions_df = pd.read_csv(datapath + "DatasetVersions.csv")[[
    "Id", "DatasetId", "DatasourceVersionId", "CreatorUserId",
    "LicenseName", "CreationDate", "Title", "Slug", "Description"]]

    kernel_languages_df = pd.read_csv(datapath + "KernelLanguages.csv")[[
    "Id", "DisplayName"
    ]]

    return kernels_df, users_df, kernel_versions_df, kernel_version_ds_df, dataset_versions_df, kernel_languages_df


