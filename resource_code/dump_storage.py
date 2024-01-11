import gzip
import shutil
import os
import config
import pandas as pd

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

def count_dump_triples(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as gzipped_file:
        line_count = sum(1 for line in gzipped_file)
    return line_count

def unzip_and_save(file_path):
    # Check if the file is a gzip file
    if not file_path.endswith('.gz'):
        raise ValueError("Not a Gzip file")

    # Define the output file path
    output_file_path = os.path.splitext(file_path)[0]

    with gzip.open(file_path, 'rt', encoding='utf-8') as gzipped_file:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(gzipped_file.read())

    return 

