########### ENDPOINTS ###########
GRAPH_DB_HOST = "http://localhost:7200"
OPEN_ML_REPOSITORY = "openml"
PWC_REPOSITORY = "pwc"
KAGGLE_REPOSITORY = "kaggle"
VIRTUOSO_HOST = "http://193.190.127.237:8890/DAV/home/dba/rdf_sink/"

########### GRAPHS ###########
GRAPH_DB_DEFAULT_GRAPH = "<http://www.ontotext.com/explicit>"
GRAPH_DB_SCHEMA = "<http://www.openrdf.org/schema/sesame#nil>"
TEST_GRAPH = "<http://193.190.127.237:8890/test>"
OPENML_DATASET_GRAPH = "<http://w3id.org/mlsea/openml/dataset>"
OPENML_RUN_GRAPH = "<http://w3id.org/mlsea/openml/run>"
OPENML_FLOW_GRAPH = "<http://w3id.org/mlsea/openml/flow>"
OPENML_TASK_GRAPH = "<http://w3id.org/mlsea/openml/task>"
PWC_GRAPH = "<http://w3id.org/mlsea/pwc>"
KAGGLE_GRAPH = "<http://w3id.org/mlsea/kaggle>"


########### QUERY VARIABLES ###########
GRAPH = "<>"
INPUT = "INPUT"


########### PREFIXES ###########

NAMESPACES = ["mlso-dc", 
              "mlso-ep", "mlso-tt", "mlso-em", "mlso",
              "rdfs", "dcterms", "mls",
              "skos", "xsd", "schema", "foaf", "prov",
              "open", "dcat", "edam", "sdo"]

URIS = ["http://w3id.org/mlso/vocab/dataset_characteristic/",
        "http://w3id.org/mlso/vocab/estimation_procedure/", 
        "http://w3id.org/mlso/vocab/ml_task_type/",
        "http://w3id.org/mlso/vocab/evaluation_measure/",
        "http://w3id.org/mlso/", 
        "http://www.w3.org/2000/01/rdf-schema#", "http://purl.org/dc/terms/",
        "http://www.w3.org/ns/mls#", "http://www.w3.org/2004/02/skos/core#",
        "http://www.w3.org/2001/XMLSchema#", "http://schema.org/",
        "http://xmlns.com/foaf/0.1/", "http://www.w3.org/ns/prov#",
        "http://open.vocab.org/terms/", "http://www.w3.org/ns/dcat#",
        "http://edamontology.org/", "https://w3id.org/okn/o/sd#"]

PREFIXES = """
PREFIX mlso: <http://w3id.org/mlso/>
PREFIX mlso-data: <http://w3id.org/mlsea/>
PREFIX mlso-dc: <http://w3id.org/mlso/vocab/dataset_characteristic/>
PREFIX mlso-ep: <http://w3id.org/mlso/vocab/estimation_procedure/>
PREFIX mlso-tt: <http://w3id.org/mlso/vocab/ml_task_type/>
PREFIX mlso-em: <http://w3id.org/mlso/vocab/evaluation_measure/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX mls: <http://www.w3.org/ns/mls#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX schema: <http://schema.org/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX open: <http://open.vocab.org/terms/>
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX edam: <http://edamontology.org/>
PREFIX sdo: <https://w3id.org/okn/o/sd#>

"""



##################
##### SELECT #####
##################

SELECT_FROM_NG = """
SELECT ?s ?p ?o
WHERE {
    GRAPH """ f"""{GRAPH}""" """ 
    {
        ?s ?p ?o.
    }
}
"""

SELECT_ID_COUNT = """
    SELECT (COUNT(?iri) AS ?instanceCount)
    WHERE {
        GRAPH """ f"""{GRAPH}""" """ 
        {
            ?iri dcterms:identifier ?id .
        }
    }
"""


##################
##### DELETE #####
##################

DEL_FROM_NG = """
WITH """ f"""{GRAPH}""" """ 
    DELETE {<http://example.org/test_named_graph/1> ?p ?o}
    WHERE {<http://example.org/test_named_graph/1> ?p ?o}
"""


##################
##### INSERT #####
##################
INSERT_TO_NG = """
WITH """ f"""{GRAPH}""" """ 
    INSERT {<http://example.com/12> a owl:Thing .}
    WHERE {}
"""

INSERT_DATA_TO_NG = """
INSERT DATA
{ GRAPH """ f"""{GRAPH}""" """
    { """ f"""{INPUT}""" """ } }
"""


##################
##### CLEAR #####
##################

CLEAR_NG = """
CLEAR GRAPH """ f"""{GRAPH}""" 


##################
##### DROP #####
##################

DROP_NG = """
DROP GRAPH """ f"""{GRAPH}""" 

def get_query(query, graph, data=None):

    if graph != None:
        query = query.replace(GRAPH, graph)
    if data != None:
        query = query.replace(INPUT, data)
    return PREFIXES + query


def add_prefix(string):

    for i in range(len(URIS)):
        if (URIS[i]) in string:
            return string.replace(URIS[i], NAMESPACES[i] + ":")

    return string

