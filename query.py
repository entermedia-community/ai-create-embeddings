import random
from typing import Dict, List

from pymilvus import Collection, DataType, connections, utility

MILVUS_URI = "http://mediadb45.entermediadb.net:19530"
COLLECTION_NAME = "client_demo_embeddings"


def _connect() -> None:
	connections.connect(alias="default", uri=MILVUS_URI)


def _get_vector_fields(collection: Collection) -> List[str]:
	return [
		field.name
		for field in collection.schema.fields
		if field.dtype in (DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR)
	]


def _get_vector_dim(collection: Collection, vector_field: str) -> int:
	for field in collection.schema.fields:
		if field.name == vector_field:
			return int(field.params.get("dim"))
	raise ValueError("Vector field not found in collection schema.")


def get_collection_stats() -> Dict[str, object]:
	_connect()

	if not utility.has_collection(COLLECTION_NAME):
		raise ValueError(f"Collection '{COLLECTION_NAME}' does not exist.")

	collection = Collection(COLLECTION_NAME)
	collection.load()

	vector_fields = _get_vector_fields(collection)
	total_rows = collection.num_entities

	vector_info = {
		field: {"dim": _get_vector_dim(collection, field), "count": total_rows}
		for field in vector_fields
	}

	return {
		"collection": COLLECTION_NAME,
		"total_rows": total_rows,
		"vector_fields": vector_info,
		"indexes": [index.to_dict() for index in collection.indexes],
	}


def quick_test_query(limit: int = 3) -> None:
	_connect()

	if not utility.has_collection(COLLECTION_NAME):
		raise ValueError(f"Collection '{COLLECTION_NAME}' does not exist.")

	collection = Collection(COLLECTION_NAME)
	collection.load()

	vector_fields = _get_vector_fields(collection)
	if not vector_fields:
		raise ValueError("No vector field found in collection schema.")

	vector_field = vector_fields[0]
	dim = _get_vector_dim(collection, vector_field)

	query_vector = [random.random() for _ in range(dim)]

	search_params = {
		"metric_type": "IP",
		"params": {"nprobe": 10},
	}

	results = collection.search(
		data=[query_vector],
		anns_field=vector_field,
		param=search_params,
		limit=limit,
	)

	print(f"Collection: {COLLECTION_NAME}")
	print(f"Vector field: {vector_field} (dim={dim})")
	print(f"Total results: {len(results[0])}")
	for hit in results[0]:
		print("-" * 40)
		print(f"ID: {hit.id}")
		print(f"Score: {hit.score}")


if __name__ == "__main__":
	stats = get_collection_stats()
	print("Collection stats:")
	for key, value in stats.items():
		print(f"{key}: {value}")

