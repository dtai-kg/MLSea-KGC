# MLSea Resource Code

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10287682.svg)](https://zenodo.org/doi/10.5281/zenodo.10287682)

<br><br>

This repository contains source code and RML mappings used for creating **MLSea-KG**, a **declaratively** constructed and regularly updated machine learning KG with more than **1.44 billion RDF triples** containing metadata about machine learning:
- Datasets 
- Tasks
- Implementations and related hyper-parameters 
- Experiment executions, their configuration settings and evaluation results 
- Code notebooks and repositories 
- Algorithms 
- Publications 
- Models 
- Scientists and practitioners 

The data were gathered and integrated from OpenML, Kaggle and Papers with Code.

<br><br>

# MLSea-KG Construction Process Overview 
![Error loading the image!](images/kgc.jpg)  

<br><br>

# Data Integration

Resource code directory contains resource code used for **collecting**, **pre-processing**, **sampling** and **declaratively generating** RDF triples, using the declarative mappings included. The input data sources used are the OpenML data extracted from the OpenML API, the Meta Kaggle CSVs and the Papers with Code dumps, which are not included in this repository.
OpenML CSV dumps are also generated, to store data retrieved from the OpenML API.

# RML Mappings

The **RML mapping** that were used for each platform are also provided, demonstrating the rules used to declaratively construct **MLSea-KG**. Both common RML mappings and the corresponding in-memory RML mappings used to generate RDF from in-memory samples are provided, complemented by their YARRRML serialization.

# Querying MLSea-KG

**MLSea-KG** is accessible through our SPARQL [endpoint](http://w3id.org/mlsea-kg). The [sparql_examples](https://github.com/dtai-kg/MLSea-Discover/tree/main/sparql_examples) folder contains example queries for traversing **MLSea-KG**. 

# MLSea-KG Snapshots

**MLSea-KG** snapshots are available at **MLSea-KG's** [Zenodo repository](https://zenodo.org/doi/10.5281/zenodo.10287349).

# Resource Code Pagkage Installation

Clone the repository:

    git clone https://github.com/dtai-kg/MLSea-Discover.git

Install dependencies: 

    pip install requirements.txt