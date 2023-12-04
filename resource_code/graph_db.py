from graphdb.rdf4j.api.repositories_api import RepositoriesApi
from graphdb.rdf4j.configuration import Configuration
from graphdb.rdf4j.api_client import ApiClient
from graphdb.rdf4j.api.graph_store_api import GraphStoreApi
from graphdb.rdf4j.api.namespaces_api import NamespacesApi
from graphdb.mime_types import RDFTypes
from SPARQLWrapper import SPARQLWrapper, JSON, POST, DIGEST, BASIC
from queries import *
import requests
from pyoxigraph import Store, NamedNode
import sys


class GraphDB_API:

    def __init__(self, host, repository):
        self.conf = Configuration()
        self.conf.host = host
        self.api_client = ApiClient(configuration=self.conf)
        self.api_client.set_default_header("Content-Type", RDFTypes.TURTLE)
        self.repository = repository

        self.repo_api = RepositoriesApi(self.api_client)
        self.graph_api = GraphStoreApi(self.api_client)
        self.names_api = NamespacesApi(self.api_client)

    def update_named_graph(self, graph):
        
        try: 
            self.graph_api.add_statements_to_indirect_namedgraph(self.repository, self.test_named_graph, 
            rdf_data="@prefix ex:<http://example.org/test_named_graph/>. \
            ex:a ex:p ex:hey. \
            ex:a ex:p ex:hey2. \
            ex:a ex:p ex:hey3.")
        except Exception as ex:
            if ex == "invalid literal for int() with base 10: ''":
                pass

        return

    def get_repositories(self):

        repositories = self.repo_api.get_all_repositories()
        print(repositories)

        return

    def get_repository_size(self):

        size = self.repo_api.get_repository_size(self.repository)
        print(size)

        return

    def set_namespaces(self):

        for i in range(len(NAMESPACES)):
            try: 
                self.names_api.set_namespace_for_prefix(self.repository, NAMESPACES[i], URIS[i])
            except Exception as ex:
                if ex == "invalid literal for int() with base 10: ''":
                    pass

        # try: 
        #     self.names_api.delete_namespace_for_prefix(self.repository, "ns1")
        # except Exception as ex:
        #     if ex == "invalid literal for int() with base 10: ''":
        #         pass
        return


class GraphDB_SW:

    def __init__(self, host, repository):
        self.host = host
        self.repository = repository
        self.readURL = host + "/repositories/" + repository 
        self.writeURL = self.readURL + "/statements"
        try:
            self.db = SPARQLWrapper(endpoint = self.readURL, updateEndpoint = self.writeURL)
        except:
            print("Database connection not completed! Returning...")
            return
        # self.sparqlUpdateStore = sparqlstore.SPARQLUpdateStore(query_endpoint=self.readURL,
        # update_endpoint=self.writeURL)

    def get_triples(self, query):
        
        self.db.setReturnFormat(JSON)
        self.db.setQuery(query)

        try:
            ret = self.db.queryAndConvert()
            return ret
        #     for r in ret["results"]["bindings"]:
        #         print(add_prefix(r['s']['value']), 
        #               add_prefix(r['p']['value']), 
        #               add_prefix(r['o']['value']))
        except Exception as e:
            print(e)
            return None


    def update(self, query):

        self.db.setHTTPAuth(DIGEST)
        self.db.setCredentials('login', 'password')
        self.db.setQuery(query)
        self.db.method = "POST"
        self.db.setReturnFormat('json')
        result = self.db.query()

        return

    def clear_graph(self, graph):

        query = get_query(CLEAR_NG, graph)
        self.update(query)

        return

    
    def insert_data(self, named_graph, rdf_graph):

        rdf_data = rdf_graph.serialize(format="turtle").encode('utf-8')
        endpoint_url = self.host + "/repositories/" + self.repository + "/statements?context=" +named_graph
        headers = {
        "Content-Type": "application/x-turtle",  # Adjust the content type as needed
        }

        response = requests.post(endpoint_url, data=rdf_data, headers=headers)

        if response.status_code == 200 or response.status_code == 204:
            print("RDF data added successfully to the specified named graph in GraphDB.")
        else:
            print(f"Error adding RDF data: {response.status_code} - {response.text}")
            print("Exiting integration...\n")
            #sys.exit()
        return response.status_code





if __name__ == "__main__":

    # flow_graph = integrate_openml(datapath, 
    #     run_batch_offset, run_batch_size,
    #     dataset_batch_offset, dataset_batch_size,
    #     task_batch_offset, task_batch_size,
    #     flow_batch_offset, flow_batch_size)

    host = GRAPH_DB_HOST
    repository = KAGGLE_REPOSITORY

    db = GraphDB_API(host, repository)
    db.set_namespaces()

    # db = GraphDB_SW(host, repository)

    # # db.insert_data(TEST_GRAPH, flow_graph)
    # db.get_triples( get_query(SELECT_FROM_NG, PWC_GRAPH))
    