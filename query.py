import random
from typing import List

from pymilvus import Collection, DataType, connections, utility

MILVUS_URI = "http://mediadb45.entermediadb.net:19530"
COLLECTION_NAME = "client_demo_embeddings"


def _connect() -> None:
	connections.connect(alias="default", uri=MILVUS_URI)


def _get_vector_field(collection: Collection) -> str:
	for field in collection.schema.fields:
		if field.dtype in (DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR):
			return field.name
	raise ValueError("No vector field found in collection schema.")


def _get_vector_dim(collection: Collection, vector_field: str) -> int:
	for field in collection.schema.fields:
		if field.name == vector_field:
			return int(field.params.get("dim"))
	raise ValueError("Vector field not found in collection schema.")


def _get_output_fields(collection: Collection, vector_field: str) -> List[str]:
	output_fields: List[str] = []
	for field in collection.schema.fields:
		if field.name == vector_field:
			continue
		if field.dtype in (DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR):
			continue
		output_fields.append(field.name)
	return output_fields


def quick_test_query(limit: int = 3) -> None:
	_connect()

	if not utility.has_collection(COLLECTION_NAME):
		raise ValueError(f"Collection '{COLLECTION_NAME}' does not exist.")

	collection = Collection(COLLECTION_NAME)
	collection.load()

	vector_field = _get_vector_field(collection)
	dim = _get_vector_dim(collection, vector_field)
	output_fields = _get_output_fields(collection, vector_field)

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
		output_fields=output_fields,
	)

	print(f"Collection: {COLLECTION_NAME}")
	print(f"Vector field: {vector_field} (dim={dim})")
	print(f"Total results: {len(results[0])}")
	for hit in results[0]:
		print("-" * 40)
		print(f"ID: {hit.id}")
		print(f"Score: {hit.score}")
		if hit.entity is not None:
			for field in output_fields:
				if field in hit.entity:
					print(f"{field}: {hit.entity[field]}")


if __name__ == "__main__":
	quick_test_query()