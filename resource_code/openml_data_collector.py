import openml
import pandas as pd
import signal
import numpy as np
import multiprocessing
import validators
import config
from preprocessing_modules import preprocess_string


def extract_run_sources(n_runs, timeout, batch_size, latest_run_id):

    filepath = config.OPENML_INPUT

    # Download OpenML Runs
    print("Extracting OpenML Run data...")
    try:
        initial_runs = openml.runs.list_runs(
            id=list(range(latest_run_id, latest_run_id + n_runs)))
        run_df = pd.DataFrame.from_dict(initial_runs, orient="index")[
            ['run_id', 'task_id', 'setup_id', 'flow_id', 'uploader',
             'task_type', 'upload_time', 'error_message']
        ].reset_index()
        initial_runs = list(run_df['run_id'])
    except Exception as e:
        print("All Runs are already collected! \
              Exception message:", str(e))
        return
    n_runs = len(initial_runs)
    print(
        f"Initial Run data acquired. Found {n_runs} new Runs. \
        Processing for metadata...")

    # Initialise lists to store more metadata
    run_list_names = ["run_desc", "run_openml_url", "run_predictions_url",
                      "run_uploader_name", "run_did",
                      "eval_per_run", "eval_keys", "eval_values",
                      "setting_per_run", "setting_flow",
                      "setting_keys", "setting_values"]
    run_dict = initialise_metadata_lists(run_list_names)

    # Process each run for metadata
    signal.signal(signal.SIGALRM, timeout_handler)
    rcounter, current_cp, first_batch = 0, 0, True
    for rid in initial_runs:
        rcounter += 1

        signal.alarm(timeout)
        try:
            run = openml.runs.get_run(rid)
            signal.alarm(0)
            if run.description_text is not None:
                run_dict["run_desc"] += preprocess_string(
                    run.description_text),
            else:
                run_dict["run_desc"] += None,
            if check_if_url(run.openml_url):
                run_dict["run_openml_url"] += run.openml_url,
            else:
                run_dict["url"] += None,
            if check_if_url(run.predictions_url):
                run_dict["run_predictions_url"] += run.predictions_url,
            else:
                run_dict["run_predictions_url"] += None,
            run_dict["run_uploader_name"] += run.uploader_name,
            run_dict["run_did"] += int(run.dataset_id),

            evaluations = run.evaluations
            for key, value in evaluations.items():
                run_dict["eval_per_run"].append(rid)
                run_dict["eval_keys"].append(key)
                run_dict["eval_values"].append(value)

            settings = run.parameter_settings
            for j in range(len(settings)):
                dict_vals = []
                run_dict["setting_per_run"].append(rid)
                run_dict["setting_flow"].append(run.flow_id)
                for key, value in settings[j].items():
                    dict_vals.append(value)
                run_dict["setting_keys"].append(dict_vals[0])
                run_dict["setting_values"].append(dict_vals[1])

        except Exception as e:
            run_dict["run_desc"] += None,
            run_dict["run_openml_url"] += None,
            run_dict["run_predictions_url"] += None,
            run_dict["run_uploader_name"] += None,
            run_dict["run_did"] += None,
            print(
                f"Skipping faulty metadata for run {rid}. \
                  Exception {e}")

        finally:
            signal.alarm(0)
            if (rcounter == batch_size) and (latest_run_id == 0):
                batch_run_df = run_df.iloc[:rcounter].copy()
                batch_run_df, evaluations_df, setting_df = \
                    populate_metadata_run_dfs(batch_run_df, run_dict)
                run_dict = initialise_metadata_lists(run_list_names)

                batch_run_df.to_csv(filepath + "runs.csv", index=False)
                evaluations_df.to_csv(
                    filepath + "run_evaluations.csv", index=False)
                setting_df.to_csv(filepath + "run_settings.csv", index=False)

                current_cp = rcounter
                first_batch = False
                print(
                    f"Runs collection initialized: {rcounter}/{n_runs} \
                    ({round(100.0*float(rcounter)/float(n_runs),1)}%)")

            elif rcounter % batch_size == 0:
                if first_batch:
                    batch_run_df = run_df.iloc[:rcounter].copy()
                else:
                    batch_run_df = run_df.iloc[(
                        rcounter-batch_size):rcounter].copy()
                batch_run_df, evaluations_df, setting_df = \
                    populate_metadata_run_dfs(batch_run_df, run_dict)
                run_dict = initialise_metadata_lists(run_list_names)

                batch_run_df.to_csv(filepath + "runs3.csv",
                                    mode='a', index=False, header=False)
                evaluations_df.to_csv(
                    filepath + "run_evaluations3.csv", mode='a',
                    index=False, header=False)
                setting_df.to_csv(filepath + "run_settings3.csv",
                                  mode='a', index=False, header=False)

                current_cp = rcounter
                first_batch = False
                print(
                    f"Runs collected: {rcounter}/{n_runs} \
                    ({round(100.0*float(rcounter)/float(n_runs),1)}%)")

    if rcounter % batch_size != 0:
        batch_run_df = run_df.iloc[current_cp:].copy()
        batch_run_df, evaluations_df, setting_df = populate_metadata_run_dfs(
            batch_run_df, run_dict)

        batch_run_df.to_csv(filepath + "runs3.csv",
                            mode='a', index=False, header=False)
        evaluations_df.to_csv(filepath + "run_evaluations3.csv",
                              mode='a', index=False, header=False)
        setting_df.to_csv(filepath + "run_settings3.csv",
                          mode='a', index=False, header=False)
        print(
            f"Runs collected: {rcounter}/{n_runs} \
            ({round(100.0*float(rcounter)/float(n_runs),1)}%)")

    print("Run extraction complete!\n")

    return


def populate_metadata_run_dfs(batch_run_df, run_dict):
    batch_run_df['description'] = run_dict["run_desc"]
    batch_run_df['openml_url'] = run_dict["run_openml_url"]
    batch_run_df['predictions_url'] = run_dict["run_predictions_url"]
    batch_run_df['uploader_name'] = run_dict["run_uploader_name"]
    batch_run_df['did'] = run_dict["run_did"]
    batch_run_df['did'] = batch_run_df['did'].astype('Int64')

    evaluations_df = pd.DataFrame()
    evaluations_df['run_id'] = run_dict["eval_per_run"]
    evaluations_df['eval_name'] = run_dict["eval_keys"]
    evaluations_df['value'] = run_dict["eval_values"]
    evaluations_df = evaluations_df.astype({"run_id": 'Int64'})

    setting_df = pd.DataFrame()
    setting_df['run_id'] = run_dict["setting_per_run"]
    setting_df['flow_id'] = run_dict["setting_flow"]
    setting_df['param'] = run_dict["setting_keys"]
    setting_df['value'] = run_dict["setting_values"]
    setting_df = setting_df.astype({"run_id": 'Int64'})
    setting_df = setting_df.astype({"flow_id": 'Int64'})

    return (batch_run_df, evaluations_df, setting_df)


def extract_dataset_sources(n_datasets, timeout, batch_size, dataset_cp):

    filepath = config.OPENML_INPUT
    # named_graph = OPENML_DATASET_GRAPH
    # dataset_cp = find_instance_count(db, named_graph)

    # Download OpenML Datasets
    print("Extracting OpenML Dataset data...")
    try:
        datasets = openml.datasets.list_datasets(
            size=n_datasets, offset=dataset_cp)
        dataset_df = pd.DataFrame.from_dict(datasets, orient="index")[
            ['did', 'name', 'version', 'uploader', 'status', 'format',
             'MajorityClassSize', 'MinorityClassSize',
             'NumberOfClasses', 'NumberOfFeatures',
             'NumberOfInstances', 'NumberOfInstancesWithMissingValues',
             'NumberOfMissingValues', 'NumberOfNumericFeatures',
             'NumberOfSymbolicFeatures']].reset_index()
        did_list = list(dataset_df['did'])
    except Exception as e:
        print("All Datasets are already collected!")
        print(f"Exception message: {str(e)}")
        return
    n_datasets = len(did_list)
    print(
        f"Initial Dataset data acquired. Found {n_datasets} new Datasets. \
            Processing for metadata...")

    # Initialise lists to store more metadata
    dataset_list_names = ["cache_format", "description", "contributor",
                          "collection_date", "upload_date",
                          "language", "licence", "url",
                          "default_target_attribute", "row_id_attribute",
                          "ignore_attribute", "version_label", "citation",
                          "visibility", "original_data_url",
                          "paper_url", "md5_checksum", "reference_did",
                          "did_creators", "creators",
                          "did_tag", "tags",
                          "did_features", "feature_name", "feature_type",
                          "feature_missing", "feature_count"]
    dataset_dict = initialise_metadata_lists(dataset_list_names)

    # Process each datast for metadata
    signal.signal(signal.SIGALRM, timeout_handler)
    dcounter, current_cp, first_batch = 0, 0, True
    for did in did_list:
        dcounter += 1

        signal.alarm(timeout)
        try:
            dataset = openml.datasets.get_dataset(did)
            signal.alarm(0)
            dataset_dict["description"] += preprocess_string(
                dataset.description),
            dataset_dict["contributor"] += dataset.contributor,
            dataset_dict["collection_date"] += dataset.collection_date,
            dataset_dict["upload_date"] += dataset.upload_date,
            dataset_dict["language"] += dataset.language,
            dataset_dict["licence"] += dataset.licence,
            dataset_dict["default_target_attribute"] += \
                dataset.default_target_attribute,
            dataset_dict["row_id_attribute"] += dataset.row_id_attribute,
            dataset_dict["ignore_attribute"] += dataset.ignore_attribute,
            dataset_dict["version_label"] += dataset.version_label,
            dataset_dict["citation"] += dataset.citation,
            dataset_dict["visibility"] += dataset.visibility,
            dataset_dict["md5_checksum"] += dataset.md5_checksum,
            dataset_dict["cache_format"] += dataset.cache_format,
            if check_if_url(dataset.url):
                dataset_dict["url"] += dataset.url,
            else:
                dataset_dict["url"] += None,
            if check_if_url(dataset.paper_url):
                dataset_dict["paper_url"] += dataset.paper_url,
                dataset_dict["reference_did"] += did,
            else:
                dataset_dict["paper_url"] += None,
                dataset_dict["reference_did"] += None,
            if check_if_url(dataset.original_data_url):
                dataset_dict["original_data_url"] += dataset.original_data_url,
            else:
                dataset_dict["original_data_url"] += None,

            if dataset.creator:
                if isinstance(dataset.creator, str):
                    for creator in dataset.creator.split(","):
                        dataset_dict["did_creators"].append(did)
                        dataset_dict["creators"].append(creator)
                else:
                    for j in range(len(dataset.creator)):
                        dataset_dict["did_creators"].append(did)
                        dataset_dict["creators"].append(dataset.creator[j])

            if dataset.tag:
                if isinstance(dataset.tag, str):
                    for tag in dataset.tag.split(","):
                        dataset_dict["did_tag"].append(did)
                        dataset_dict["tags"].append(tag)
                elif isinstance(dataset.tag, list):
                    for j in range(len(dataset.tag)):
                        dataset_dict["did_tag"].append(did)
                        dataset_dict["tags"].append(dataset.tag[j])

            features = dataset.features
            for f in range(len(features)):
                dataset_dict["feature_count"].append(features[f].index)
                dataset_dict["feature_type"].append(features[f].data_type)
                dataset_dict["did_features"].append(did)
                dataset_dict["feature_name"].append(features[f].name)
                dataset_dict["feature_missing"].append(
                    features[f].number_missing_values)

        except Exception as e:
            dataset_dict["description"] += None,
            dataset_dict["contributor"] += None,
            dataset_dict["collection_date"] += None,
            dataset_dict["upload_date"] += None,
            dataset_dict["language"] += None,
            dataset_dict["licence"] += None,
            dataset_dict["url"] += None,
            dataset_dict["default_target_attribute"] += None,
            dataset_dict["row_id_attribute"] += None,
            dataset_dict["ignore_attribute"] += None,
            dataset_dict["version_label"] += None,
            dataset_dict["citation"] += None,
            dataset_dict["visibility"] += None,
            dataset_dict["original_data_url"] += None,
            dataset_dict["paper_url"] += None,
            dataset_dict["md5_checksum"] += None,
            dataset_dict["cache_format"] += None,
            print(
                f"Skipping faulty metadata for dataset {did}. \
                  Exception {e}")

        finally:
            signal.alarm(0)
            if (dcounter == batch_size) and (dataset_cp == 0):
                batch_dataset_df = dataset_df.iloc[:dcounter].copy()
                batch_dataset_df, creator_df, tag_df, \
                    feature_df, reference_df = (
                        populate_metadata_dataset_dfs(batch_dataset_df,
                                                      dataset_dict))
                dataset_dict = initialise_metadata_lists(dataset_list_names)

                batch_dataset_df.to_csv(filepath + "datasets.csv", index=False)
                creator_df.to_csv(
                    filepath + "dataset_creators.csv", index=False)
                tag_df.to_csv(filepath + "dataset_tags.csv", index=False)
                feature_df.to_csv(
                    filepath + "dataset_features.csv", index=False)
                reference_df.to_csv(
                    filepath + "dataset-references.csv", index=False)

                # integrate_openml_datasets(batch_dataset_df,
                # creator_df, tag_df,
                # feature_df, reference_df, db, named_graph)

                current_cp = dcounter
                first_batch = False
                print(
                    f"Datasets collection initialized: \
                    {dcounter}/{n_datasets} \
                    ({round(100.0*float(dcounter)/float(n_datasets),1)}%)")

            elif dcounter % batch_size == 0:
                if first_batch:
                    batch_dataset_df = dataset_df.iloc[:dcounter].copy()
                else:
                    batch_dataset_df = dataset_df.iloc[(
                        dcounter-batch_size):dcounter].copy()

                batch_dataset_df, creator_df, tag_df, \
                    feature_df, reference_df = (
                        populate_metadata_dataset_dfs(batch_dataset_df,
                                                      dataset_dict))
                dataset_dict = initialise_metadata_lists(dataset_list_names)

                batch_dataset_df.to_csv(
                    filepath + "datasets.csv", index=False,
                    mode='a', header=False)
                creator_df.to_csv(filepath + "dataset_creators.csv",
                                  index=False, mode='a', header=False)
                tag_df.to_csv(filepath + "dataset_tags.csv",
                              index=False, mode='a', header=False)
                feature_df.to_csv(filepath + "dataset_features.csv",
                                  index=False, mode='a', header=False)
                reference_df.to_csv(
                    filepath + "dataset_references.csv", index=False,
                    mode='a', header=False)

                # integrate_openml_datasets(batch_dataset_df,
                # creator_df, tag_df,
                # feature_df, reference_df, db, named_graph)

                current_cp = dcounter
                first_batch = False
                print(
                    f"Datasets collected: \
                        {dcounter}/{n_datasets} \
                        ({round(100.0*float(dcounter)/float(n_datasets),1)}%)")

    if dcounter % batch_size != 0:
        batch_dataset_df = dataset_df.iloc[current_cp:].copy()
        batch_dataset_df, creator_df, tag_df, feature_df, reference_df = (
            populate_metadata_dataset_dfs(batch_dataset_df, dataset_dict))

        batch_dataset_df.to_csv(filepath + "datasets.csv",
                                index=False, mode='a', header=False)
        creator_df.to_csv(filepath + "dataset_creators.csv",
                          index=False, mode='a', header=False)
        tag_df.to_csv(filepath + "dataset_tags.csv",
                      index=False, mode='a', header=False)
        feature_df.to_csv(filepath + "dataset_features.csv",
                          index=False, mode='a', header=False)
        reference_df.to_csv(filepath + "dataset_references.csv",
                            index=False, mode='a', header=False)

        # integrate_openml_datasets(batch_dataset_df.fillna(value=np.nan),
        # creator_df.fillna(value=np.nan), tag_df.fillna(value=np.nan),
        #         feature_df.fillna(value=np.nan),
        # reference_df.fillna(value=np.nan), db, named_graph)

        print(f"Datasets collected: \
             {dcounter}/{n_datasets} \
             ({round(100.0*float(dcounter)/float(n_datasets),1)}%)")

    print("Dataset extraction complete!\n")

    return


def populate_metadata_dataset_dfs(batch_dataset_df, dataset_dict):
    batch_dataset_df['cache_format'] = dataset_dict["cache_format"]
    batch_dataset_df['description'] = dataset_dict["description"]
    batch_dataset_df['contributor'] = dataset_dict["contributor"]
    batch_dataset_df['collection_date'] = dataset_dict["collection_date"]
    batch_dataset_df['upload_date'] = dataset_dict["upload_date"]
    batch_dataset_df['language'] = dataset_dict["language"]
    batch_dataset_df['licence'] = dataset_dict["licence"]
    batch_dataset_df['url'] = dataset_dict["url"]
    batch_dataset_df['default_target_attribute'] = \
        dataset_dict["default_target_attribute"]
    batch_dataset_df['row_id_attribute'] = dataset_dict["row_id_attribute"]
    batch_dataset_df['ignore_attribute'] = dataset_dict["ignore_attribute"]
    batch_dataset_df['version_label'] = dataset_dict["version_label"]
    batch_dataset_df['citation'] = dataset_dict["citation"]
    batch_dataset_df['visibility'] = dataset_dict["visibility"]
    batch_dataset_df['original_data_url'] = dataset_dict["original_data_url"]
    batch_dataset_df['paper_url'] = dataset_dict["paper_url"]
    batch_dataset_df['md5_checksum'] = dataset_dict["md5_checksum"]

    creator_df = pd.DataFrame()
    creator_df['did_per_creator'] = dataset_dict["did_creators"]
    creator_df['creator_per_dataset'] = dataset_dict["creators"]

    tag_df = pd.DataFrame()
    tag_df['did_per_tag'] = dataset_dict["did_tag"]
    tag_df['tag_per_dataset'] = dataset_dict["tags"]

    feature_df = pd.DataFrame()
    feature_df['dataset_id'] = dataset_dict["did_features"]
    feature_df['feature_id'] = dataset_dict["feature_count"]
    feature_df['name'] = dataset_dict["feature_name"]
    feature_df['type'] = dataset_dict["feature_type"]
    feature_df['missing_values'] = dataset_dict["feature_missing"]

    reference_df = pd.DataFrame()
    reference_df["did"] = dataset_dict["reference_did"]
    reference_df["paper_url"] = dataset_dict["paper_url"]
    reference_df = reference_df.dropna(axis=0)

    return (batch_dataset_df, creator_df, tag_df, feature_df, reference_df)


def extract_task_sources(n_tasks, timeout, batch_size, task_cp):

    filepath = config.OPENML_INPUT

    # Download OpenML Tasks
    print("Extracting OpenML Task data...")
    try:
        tlist = openml.tasks.list_tasks(size=n_tasks, offset=task_cp)
        task_df = pd.DataFrame.from_dict(tlist, orient="index").reset_index()[
            ['tid', 'did', 'name', 'task_type',
             'status', 'estimation_procedure',
             'evaluation_measures']]
        tlist = list(task_df['tid'])
    except Exception as e:
        print("All Tasks are already collected! \
              Exception message:", str(e))
        return
    n_tasks = len(tlist)
    print(
        f"Initial Task data acquired. Found {n_tasks} new Tasks. \
        Processing for metadata...")

    # Initialise lists to store more metadata
    task_list_names = ["estimation_procedure_type", "url"]
    task_dict = initialise_metadata_lists(task_list_names)

    signal.signal(signal.SIGALRM, timeout_handler)
    tcounter, current_cp, first_batch = 0, 0, True
    for tid in tlist:
        tcounter += 1

        signal.alarm(timeout)
        try:
            task = openml.tasks.get_task(tid)
            signal.alarm(0)
            task_dict["estimation_procedure_type"] += \
                task.estimation_procedure['type'],
            if check_if_url(task.openml_url):
                task_dict["url"] += task.openml_url,
            else:
                task_dict["url"] += None,

        except Exception as e:
            task_dict["estimation_procedure_type"] += None,
            task_dict["url"] += None,
            print(f"Skipping faulty metadata for task {tid}. \
                  Exception: {str(e)}")

        finally:
            signal.alarm(0)
            if (tcounter == batch_size) and (task_cp == 0):
                batch_task_df = task_df.iloc[:tcounter].copy()
                batch_task_df = populate_metadata_task_dfs(
                    batch_task_df, task_dict)
                task_dict = initialise_metadata_lists(task_list_names)

                batch_task_df.to_csv(filepath + "tasks.csv", index=False)

                current_cp = tcounter
                first_batch = False
                print(
                    f"Tasks collection initialized: \
                        {tcounter}/{n_tasks} \
                        ({round(100.0*float(tcounter)/float(n_tasks),1)}%)")

            elif tcounter % batch_size == 0:
                if first_batch:
                    batch_task_df = task_df.iloc[:tcounter].copy()
                else:
                    batch_task_df = task_df.iloc[(
                        tcounter-batch_size):tcounter].copy()
                batch_task_df = populate_metadata_task_dfs(
                    batch_task_df, task_dict)
                task_dict = initialise_metadata_lists(task_list_names)

                batch_task_df.to_csv(filepath + "tasks.csv",
                                     index=False, mode='a', header=False)

                current_cp = tcounter
                first_batch = False
                print(
                    f"Tasks collected: \
                    {tcounter}/{n_tasks}\
                    ({round(100.0*float(tcounter)/float(n_tasks), 1)}%)")

    if tcounter % batch_size != 0:
        batch_task_df = task_df.iloc[current_cp:].copy()
        batch_task_df = populate_metadata_task_dfs(batch_task_df, task_dict)

        batch_task_df.to_csv(filepath + "tasks.csv",
                             index=False, mode='a', header=False)

    print(
        f"Tasks collected: \
        {tcounter}/{n_tasks} \
        ({round(100.0*float(tcounter)/float(n_tasks),1)}%)")
    print("Task extraction complete!\n")

    return


def populate_metadata_task_dfs(batch_task_df, task_dict):
    batch_task_df['estimation_procedure_type'] = \
        task_dict["estimation_procedure_type"]
    batch_task_df['url'] = task_dict["url"]

    return batch_task_df


def extract_flow_sources(n_flows, timeout, batch_size, flow_cp):

    filepath = config.OPENML_INPUT

    # Download OpenML Flows
    print("Extracting OpenML Flow data...")
    try:
        flist = openml.flows.list_flows(size=n_flows, offset=flow_cp)
        flow_df = pd.DataFrame.from_dict(flist, orient="index")[
            ['id', 'full_name', 'name', 'version',
             'external_version', 'uploader']].reset_index()
        flist = list(flow_df['id'])
    except Exception as e:
        print("All Flows are already collected!")
        print(f"Exception message: {str(e)}")
        return
    n_flows = len(flist)
    print(
        f"Initial Flow data acquired. Found {n_flows} new Flows. \
        Processing for metadata...")

    # Initialise lists to store more metadata
    flow_list_names = ["dependencies", "description", "language",
                       "openml_url", "upload_date",
                       "param_flow_id", "parameter_keys",
                       "parameter_values", "parameter_desc",
                       "parameter_datatype", "tag_flow_id", "tags",
                       "dep_flow_id", "dependency"]
    flow_dict = initialise_metadata_lists(flow_list_names)

    signal.signal(signal.SIGALRM, timeout_handler)
    fcounter, current_cp, first_batch = 0, 0, True
    for fid in flist:
        fcounter += 1

        signal.alarm(timeout)
        try:
            flow = openml.flows.get_flow(fid)
            signal.alarm(0)
            flow_dict["description"] += preprocess_string(flow.description),
            flow_dict["language"] += flow.language,
            if check_if_url(flow.openml_url):
                flow_dict["openml_url"] += flow.openml_url,
            else:
                flow_dict["openml_url"] += None,
            flow_dict["upload_date"] += flow.upload_date,

            parameters = flow.parameters
            for key, value in parameters.items():
                flow_dict["param_flow_id"].append(fid)
                flow_dict["parameter_keys"].append(key)
                flow_dict["parameter_values"].append(value)
                hyperparameter_desc = \
                    flow.parameters_meta_info[key]['description']
                if hyperparameter_desc:
                    flow_dict["parameter_desc"].append(
                        preprocess_string(hyperparameter_desc))
                else:
                    flow_dict["parameter_desc"].append(None)
                flow_dict["parameter_datatype"].append(
                    flow.parameters_meta_info[key]['data_type'])

            if flow.tags:
                if isinstance(flow.tags, str):
                    for tag in flow.tags.split(","):
                        flow_dict["tag_flow_id"].append(fid)
                        flow_dict["tags"].append(tag)
                else:
                    for j in range(len(flow.tags)):
                        flow_dict["tag_flow_id"].append(fid)
                        flow_dict["tags"].append(flow.tags[j])

            if flow.dependencies:
                if "," in str(flow.dependencies):
                    dependencies = str(flow.dependencies).split(',')
                else:
                    dependencies = str(flow.dependencies).split('\n')
                for dep in dependencies:
                    if dep != "nan":
                        flow_dict["dep_flow_id"].append(fid)
                        flow_dict["dependency"].append(dep)

        except Exception as e:
            flow_dict["dependencies"] += None,
            flow_dict["description"] += None,
            flow_dict["language"] += None,
            flow_dict["openml_url"] += None,
            flow_dict["upload_date"] += None,
            print(f"Skipping faulty metadata for flow {fid}. \
                  Exception: {str(e)}")

        finally:
            signal.alarm(0)
            if (fcounter == batch_size) and (flow_cp == 0):
                batch_flow_df = flow_df.iloc[:fcounter].copy()
                batch_flow_df, param_df, \
                    flow_tag_df, dependency_df = populate_metadata_flow_dfs(
                        batch_flow_df, flow_dict)
                flow_dict = initialise_metadata_lists(flow_list_names)

                batch_flow_df.to_csv(filepath + "flows.csv", index=False)
                param_df.to_csv(filepath + "flow_params.csv", index=False)
                flow_tag_df.to_csv(filepath + "flow_tags.csv", index=False)
                dependency_df.to_csv(
                    filepath + "flow_dependencies.csv", index=False)

                current_cp = fcounter
                first_batch = False
                print(
                    f"Flows collection initialized: \
                    {fcounter}/{n_flows} \
                    ({round(100.0*float(fcounter)/float(n_flows),1)}%)")

            elif fcounter % batch_size == 0:
                if first_batch:
                    batch_flow_df = flow_df.iloc[:fcounter].copy()
                else:
                    batch_flow_df = flow_df.iloc[(
                        fcounter-batch_size):fcounter].copy()
                batch_flow_df, param_df, \
                    flow_tag_df, dependency_df = populate_metadata_flow_dfs(
                        batch_flow_df, flow_dict)
                flow_dict = initialise_metadata_lists(flow_list_names)

                batch_flow_df.to_csv(filepath + "flows.csv",
                                     index=False, mode='a', header=False)
                param_df.to_csv(filepath + "flow_params.csv",
                                index=False, mode='a', header=False)
                flow_tag_df.to_csv(filepath + "flow_tags.csv",
                                   index=False, mode='a', header=False)
                dependency_df.to_csv(
                    filepath + "flow_dependencies.csv", index=False,
                    mode='a', header=False)

                current_cp = fcounter
                first_batch = False
                print(
                    f"Flows collected: \
                    {fcounter}/{n_flows} \
                    ({round(100.0*float(fcounter)/float(n_flows),1)}%)")

    if fcounter % batch_size != 0:
        batch_flow_df = flow_df.iloc[current_cp:].copy()
        batch_flow_df, param_df, flow_tag_df, dependency_df = \
            populate_metadata_flow_dfs(
                batch_flow_df,
                flow_dict)

        batch_flow_df.to_csv(filepath + "flows.csv",
                             index=False, mode='a', header=False)
        param_df.to_csv(filepath + "flow_params.csv",
                        index=False, mode='a', header=False)
        flow_tag_df.to_csv(filepath + "flow_tags.csv",
                           index=False, mode='a', header=False)
        dependency_df.to_csv(filepath + "flow_dependencies.csv",
                             index=False, mode='a', header=False)

        print(
            f"Flows collected: \
            {fcounter}/{n_flows} \
            ({round(100.0*float(fcounter)/float(n_flows),1)}%)")

    print("Flow extraction complete!\n")

    return


def populate_metadata_flow_dfs(batch_flow_df, flow_dict):
    batch_flow_df['description'] = flow_dict["description"]
    batch_flow_df['language'] = flow_dict["language"]
    batch_flow_df['openml_url'] = flow_dict["openml_url"]
    batch_flow_df['upload_date'] = flow_dict["upload_date"]
    # batch_flow_df = batch_flow_df.drop('external_version', axis=1)
    # batch_flow_df = batch_flow_df.drop('dependencies', axis=1)
    # batch_flow_df = batch_flow_df.drop('description', axis=1)

    param_df = pd.DataFrame()
    param_df['flow_id'] = flow_dict["param_flow_id"]
    param_df['key'] = flow_dict["parameter_keys"]
    param_df['value'] = flow_dict["parameter_values"]
    param_df['description'] = flow_dict["parameter_desc"]
    param_df['datatype'] = flow_dict["parameter_datatype"]

    flow_tag_df = pd.DataFrame()
    flow_tag_df['flow_id'] = flow_dict["tag_flow_id"]
    flow_tag_df['tag'] = flow_dict["tags"]

    dependency_df = pd.DataFrame()
    dependency_df["flow_id"] = flow_dict["dep_flow_id"]
    dependency_df["dependency"] = flow_dict["dependency"]

    return (batch_flow_df, param_df, flow_tag_df, dependency_df)


def timeout_handler(num, stack):
    # print("Data download timed out!")
    raise Exception("OPENML_TIMEOUT")


def initialise_metadata_lists(list_names):
    list_dict = {name: [] for name in list_names}
    return list_dict


def get_checkpoints(filenames: list):

    checkpoints = [0, 0, 0, 0]
    latest_ids = [0, 0, 0, 0]
    for i in range(len(filenames)):
        try:
            df = pd.read_csv(filenames[i])
            checkpoints[i] = len(df.index)
            latest_ids[i] = str(df.iloc[-1].to_list()[0])
        except Exception as e:
            print(f"Could not read checkpoint file {filenames[i]}. \
                  Exception: {str(e)}")
            pass

    return checkpoints, latest_ids


def check_if_url(string):

    if validators.url(string):
        return True
    else:
        return False


def remove_badly_formed_urls(string):
    if (not check_if_url(string)) and \
            (string is not np.nan):
        return np.nan
    else:
        return string


def openml_data_collector():

    # OpenML Capacity
    # Runs: 10.1 M (10,103,426)
    # Datasets: 5.3 K (5,306)
    # Tasks: 261.4 K (261,404)
    # Flows: 16.7 K (16,719)

    print("OpenML Data Collector initiated...\n")

    openml.config.apikey = 'eee9181dd538cb1a9daac582a55efd72'
    filepath = config.OPENML_INPUT

    # host = GRAPH_DB_HOST
    # repository = OPEN_ML_REPOSITORY
    # db = GraphDB_SW(host, repository)

    run_timeout, dataset_timeout, task_timeout, flow_timeout = 2, 70, 2, 2
    batch_size = 100
    dataset_batch_size = 1

    runs_csv = filepath + "runs3.csv"
    datasets_csv = filepath + "datasets.csv"
    tasks_csv = filepath + "tasks.csv"
    flows_csv = filepath + "flows.csv"

    checkpoints, latest_ids = get_checkpoints(
        [runs_csv, datasets_csv, tasks_csv, flows_csv])
    run_cp, dataset_cp, task_cp, flow_cp = checkpoints[
        0], checkpoints[1], checkpoints[2], checkpoints[3]
    config.update_openml_checkpoints(run_cp, dataset_cp, task_cp, flow_cp)
    run_lid, dataset_lid, task_lid, flow_lid = latest_ids[
        0], latest_ids[1], latest_ids[2], latest_ids[3]

    print(
        f"Currently already collected: \
            {config.OPENML_RUN_CURRENT_OFFSET + run_cp} Runs, \
            {dataset_cp} Datasets, {task_cp} Tasks and {flow_cp} Flows")
    print(f"Latest Run collected: Run {run_lid}")
    print(f"Latest Dataset collected: Dataset {dataset_lid}")
    print(f"Latest Task collected: Task {task_lid}")
    print(f"Latest Flow collected: Flow {flow_lid}")

    print("Now scanning for new data...\n")

    # Single-thread collector
    def run_single_thread_collector():
        # print("Running single-thread collector...")
        datasets_max_search_size = 1000
        extract_dataset_sources(datasets_max_search_size,
                                dataset_timeout, dataset_batch_size,
                                dataset_cp)
        tasks_max_search_size = 100
        extract_task_sources(tasks_max_search_size,
                             task_timeout, batch_size, task_cp)
        flows_max_search_size = 100
        extract_flow_sources(flows_max_search_size,
                             flow_timeout, batch_size, flow_cp)
        runs_max_search_size = 10000
        extract_run_sources(runs_max_search_size, run_timeout,
                            batch_size,
                            config.OPENML_RUN_CURRENT_OFFSET + run_cp)
        runs_max_search_size = 100
        extract_run_sources(runs_max_search_size, run_timeout,
                            batch_size,
                            int(run_lid))  # Using run last
        # id to avoid timeouts of large offsets

    # Multi-thread collector
    def run_multi_thread_collector():
        print("Running multi-thread collector...")
        processes = []

        runs_max_search_size = 100000
        datasets_max_search_size = 1000
        tasks_max_search_size = 10000
        flows_max_search_size = 3500
        run_args = (runs_max_search_size, run_timeout, batch_size,
                    config.OPENML_RUN_CURRENT_OFFSET + run_cp)
        dataset_args = (datasets_max_search_size,
                        dataset_timeout, dataset_batch_size, dataset_cp)
        task_args = (tasks_max_search_size, task_timeout, batch_size, task_cp)
        flow_args = (flows_max_search_size, flow_timeout, batch_size, flow_cp)

        functions = [extract_run_sources, extract_dataset_sources,
                     extract_task_sources, extract_flow_sources]
        arguments = [run_args, dataset_args, task_args, flow_args]

        for func, args in zip(functions, arguments):
            process = multiprocessing.Process(target=func, args=args)
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

    # run_multi_thread_collector()
    run_single_thread_collector()

    print("Data sources collected succesfully!")
    return


if __name__ == "__main__":

    openml_data_collector()
