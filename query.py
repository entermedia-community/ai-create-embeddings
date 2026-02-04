from pymilvus import MilvusClient

MILVUS_URI = "http://mediadb45.entermediadb.net:19530"
# MILVUS_URI = "./mil.db"
COLLECTION_NAME = "client_demo_embeddings"

client = MilvusClient(uri=MILVUS_URI)

# Print available collections
try:
    collections = client.list_collections()
except Exception:
    collections = []
print("available_collections:", collections)

res = client.query(
  collection_name=COLLECTION_NAME,
  limit=10,
  output_fields=["parent_id", "text"],
)

print(res)

client.close()