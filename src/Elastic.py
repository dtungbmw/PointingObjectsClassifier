from elasticsearch import Elasticsearch
from datetime import datetime



class ElasticsearchClient:
    def __init__(self, host='localhost', port=9200, username='elastic'):
        """
        Initialize the connection to the Elasticsearch instance.

        :param host: Elasticsearch host (default is localhost)
        :param port: Elasticsearch port (default is 9200)
        """
        CERT_FINGERPRINT="73:ED:97:90:E9:A9:64:96:66:61:D6:18:16:47:A8:1E:F4:30:EF:D6:40:55:1F:C1:73:1A:FE:19:2F:6A:10:C7"
        ELASTIC_PASSWORD = 'PV13_NIq1rsf9rixIaie'
        self.es = Elasticsearch(
            "https://localhost:9200",
            ssl_assert_fingerprint=CERT_FINGERPRINT,
            basic_auth=("elastic", ELASTIC_PASSWORD)
        )
        if self.es.ping():
            print("Connected to Elasticsearch")
        else:
            print("Could not connect to Elasticsearch")
            raise ConnectionError("Failed to connect to Elasticsearch")

    def insert_data(self, index_name, document_id, data):
        """
        Inserts data into the specified Elasticsearch index.

        :param index_name: Name of the Elasticsearch index
        :param document_id: ID of the document to insert
        :param data: Data to be inserted (must be a dictionary)
        :return: Response from Elasticsearch
        """
        try:
            response = self.es.index(index=index_name, id=document_id, body=data)
            print(f"Data inserted with ID {document_id}: {response}")
            return response
        except Exception as e:
            print(f"Error indexing data: {e}")
            return None

    def get_data(self, index_name, document_id):
        """
        Retrieves data from the specified Elasticsearch index.

        :param index_name: Name of the Elasticsearch index
        :param document_id: ID of the document to retrieve
        :return: Retrieved document data
        """
        try:
            response = self.es.get(index=index_name, id=document_id)
            print(f"Retrieved document with ID {document_id}: {response['_source']}")
            return response['_source']
        except Exception as e:
            print(f"Error retrieving data: {e}")
            return None

    def refresh_index(self, index_name):
        """
        Refreshes the specified index, making documents immediately searchable.

        :param index_name: Name of the Elasticsearch index
        """
        try:
            self.es.indices.refresh(index=index_name)
            print(f"Index '{index_name}' refreshed.")
        except Exception as e:
            print(f"Error refreshing index: {e}")
