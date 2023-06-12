import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import torch
import re
import time


def build_collection(history, character):
	
	start_time = time.time()

	chroma_client = chromadb.Client(Settings(anonymized_telemetry=False, chroma_db_impl="duckdb+parquet", persist_directory=f"embeddings/{character}"))

	sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
	collection = chroma_client.create_collection(name="context", embedding_function=sentence_transformer_ef)

	chunks = []
	hist_size = len(history['You'])
	for i in range(hist_size-1):
		chunks.append(make_single_exchange(i, history, character))

	ids = [f"id{i}" for i in range(len(chunks))]
	collection.add(documents=chunks, ids=ids)

	collection_creation_time = time.time()
	print("--- %s seconds ---" % (collection_creation_time - start_time))

	chroma_client.persist()

	return collection


def custom_generate_chat_prompt(user_input, collection, history, character, params):
	if len(history['You']) > params['chunk_count'] and user_input != '':

		query = user_input

		best_ids = get_ids_sorted(ids, collection, query, n_results=params['chunk_count'], n_initial=params['chunk_count_initial'], time_weight=params['time_weight'])
		additional_context = '\n'

		for id_ in best_ids:
			additional_context += make_single_exchange(id_, history, character)

		return additional_context

	else:
		return ''


def make_single_exchange(id_, history, character):
    output = ''
    output += f"You: {history['You'][id_]}\n"
    output += f"{character}: {history[character][id_]}\n"
    return output


def get_ids_sorted(ids, collection, search_strings, n_results, n_initial: int = None, time_weight: float = 1.0):
	do_time_weight = time_weight > 0
	if not (do_time_weight and n_initial is not None):
		n_initial = n_results
	elif n_initial == -1:
		n_initial = len(ids)

	if n_initial < n_results:
		raise ValueError(f"n_initial {n_initial} should be >= n_results {n_results}")

	_, new_ids, distances = get_documents_ids_distances(ids, collection, search_strings, n_initial)
	if do_time_weight:
		distances_w = apply_time_weight_to_distances(len(ids), new_ids, distances, time_weight=time_weight)
		results = zip(new_ids, distances, distances_w)
		results = sorted(results, key=lambda x: x[2])[:n_results]
		results = sorted(results, key=lambda x: x[0])
		ids = [x[0] for x in results]

	return sorted(ids)


def get_documents_ids_distances(ids, collection, search_strings, n_results):
	n_results = min(len(ids), n_results)
	if n_results == 0:
		return [], [], []

	result = collection.query(query_texts=search_strings, n_results=n_results, include=['documents', 'distances'])
	documents = result['documents'][0]
	new_ids = list(map(lambda x: int(x[2:]), result['ids'][0]))
	distances = result['distances'][0]
	return documents, new_ids, distances


def apply_time_weight_to_distances(collection_length, ids, distances, time_weight: float = 1.0):
	return [distance * (1 - _id / (collection_length - 1) * time_weight) for _id, distance in zip(ids, distances)]