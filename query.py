from pymilvus import MilvusClient

MILVUS_URI = "http://mediadb45.entermediadb.net:19530"
COLLECTION_NAME = "client_demo_embeddings"

client = MilvusClient(uri=MILVUS_URI)
res = client.query(
  collection_name=COLLECTION_NAME,
  limit=10,
)

print(res)

client.close()