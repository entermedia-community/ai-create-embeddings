from pymilvus import MilvusClient

MILVUS_URI = "http://mediadb45.entermediadb.net:19530"
COLLECTION_NAME = "client_demo_embeddings"

client = MilvusClient(uri=MILVUS_URI)
res = client.query(
  collection_name=COLLECTION_NAME,
  filter="parent_id == 'entityasset_AZu-72vdIKT_Qmrri3W9'",
)

print(res)

total_rows = client.count_entities(collection_name=COLLECTION_NAME)
print({"total_rows": total_rows})

client.close()