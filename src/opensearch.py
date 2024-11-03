from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
import os

# AWS Region
region = 'us-east-1'  # Change this to your AWS region

# OpenSearch endpoint (domain)
host = 'vpc-dev-iedata-es-tset-7e63ja2iyhtzwjntle47vrpdam.us-east-1.es.amazonaws.com'  # Replace with your OpenSearch endpoint


#[563969369487_PowerUserAccess]
access_key_id=os.getenv('AWS_ACCESS_KEY_ID')
secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
session_token=os.getenv('AWS_SESSION_TOKEN')
# Credentials for AWS authentication
service = 'es'
#credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(access_key_id, secret_access_key, None, region, service)

# Sample data to be indexed
data = {
    "title": "OpenSearch with Python",
    "description": "Saving data to OpenSearch from an EC2 instance using Python.",
    "author": "Your Name",
    "date": "2024-09-13"
}

# Index data into OpenSearch
index_name = 'pointing-eval'  # Replace with your index name
document_id = '1'  # You can auto-generate or specify a unique document ID


def save_data():
    # Initialize the OpenSearch client
    client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )

    response = client.index(
        index=index_name,
        id=document_id,
        body=data
    )

    print(response)
