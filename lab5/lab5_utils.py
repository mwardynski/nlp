from datasets import load_dataset
import json
import math
import requests

def load_fiqa():
    corpus = load_dataset("clarin-knext/fiqa-pl", name="corpus")
    queries = load_dataset("clarin-knext/fiqa-pl", name="queries")
    qrels = load_dataset("clarin-knext/fiqa-pl-qrels")
    return corpus, queries, qrels

fts_url = "http://localhost:9200"
fiqa_index_settings = {
    "settings": {
        "analysis": {
            "filter": {
                "polish_morfologik": {
                    "type": "morfologik_stem"
                }
            },
            "analyzer": {
                "polish_analyzer_morf": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "polish_morfologik",
                        "lowercase"
                    ]
                }
            }
        }
    },
    "mappings": {
        "properties": {
        "text": {
            "type": "text",
            "analyzer": "polish_analyzer_morf"
            }
        }
    }
}

def create_fts_index(index_name):
    index_url = F"{fts_url}/{index_name}"
    delete_response = requests.delete(f"{index_url}")
    if delete_response.status_code == 200:
        print(f"Index '{index_name}' deleted successfully.")
    else:
        print(f"Failed to delete index '{index_name}': {delete_response.text}")

    response = requests.put(index_url, headers={"Content-Type": "application/json"}, data=json.dumps(fiqa_index_settings))
    if response.status_code == 200:
        print("Index created.")
    else:
        print(f"Index creation failed: {response.text}")

    return fts_url, index_url

def bulk_load(fts_url, index_name, dict):
    bulk_data = ""
    for key, value in dict.items():
        doc_id = key
        bulk_data += json.dumps({"index": {"_index": index_name, "_id": doc_id}}) + "\n"
        bulk_data += json.dumps({"text": value}) + "\n"



    bulk_response = requests.post(f"{fts_url}/_bulk", headers={"Content-Type": "application/x-ndjson"}, data=bulk_data)

    if bulk_response.status_code == 200:
        response_data = bulk_response.json()
        if any(item.get("index", {}).get("error") for item in response_data["items"]):
            print("Some documents failed to index:")
            for item in response_data["items"]:
                if "error" in item["index"]:
                    print(item["index"]["error"])
        else:
            print("All documents indexed successfully.")
    else:
        print(f"Failed to index data: {bulk_response.text}")

def prepare_fiqa_qrels(fiqa_qrels, subsets):
    query_to_corpus_dict = {}

    for subset in subsets:
        for item in fiqa_qrels[subset]:
            if item['query-id'] not in query_to_corpus_dict:
                query_to_corpus_dict[item['query-id']] = {}

            query_to_corpus_dict[item['query-id']][item['corpus-id']] = item['score']

    for query_id in query_to_corpus_dict:
        sorted_corpuses_by_score = dict(sorted(query_to_corpus_dict[query_id].items(), key=lambda item: item[1]))
        query_to_corpus_dict[query_id] = sorted_corpuses_by_score

    return query_to_corpus_dict

def prepare_fiqa_corpus_related_to_selected_subsets(fiqa_corpus, query_to_corpus_dict, in_subsets):
    avoid_corpus_set = set()
    corpus_dict = {}
    
    for query_id in query_to_corpus_dict.keys():
        for corpus_id in query_to_corpus_dict[query_id]:
            avoid_corpus_set.add(int(corpus_id))

    corpus = fiqa_corpus["corpus"]
    for entry in corpus:
        if (int(entry['_id']) in avoid_corpus_set) == in_subsets:
            corpus_dict[int(entry['_id'])] = entry['text']

    return corpus_dict

def prepare_fiqa_queries_for_selected_subset(fiqa_queries, query_to_corpus_dict):
    queries_dict = {}

    queries_dataset = fiqa_queries['queries']
    for entry in queries_dataset:
        if int(entry['_id']) in query_to_corpus_dict.keys():
            queries_dict[int(entry['_id'])] = entry['text']

    return queries_dict

def find_for_phrase_with_exclusion(index_url, search_phrase, search_field, size, excluded_ids):
    search_query = {
        "size": size,
        "query": {
            "bool": {
                "must": {
                    "match": {
                        search_field: {
                            "query": search_phrase,
                        }
                    }
                },
                "must_not": {
                    "ids": {
                        "values": excluded_ids
                    }
                }
            }
        }
    }

    response = requests.get(f"{index_url}/_search", headers={"Content-Type": "application/json"}, data=json.dumps(search_query))

    if response.status_code == 200:
        search_results = response.json()
        return dict(list(map(lambda hit: (int(hit['_id']), float(hit['_score'])), search_results["hits"]["hits"])))
    else:
        print(f"Search failed: {response.text}")


relevant_doc_number = 0
relevant_docs = []

def calculate_dcg(docs, docs_scoring):
    sum = 0
    for i, doc_id in enumerate(docs):
        if doc_id in docs_scoring.keys():
            sum += (2**docs_scoring[doc_id]-1)/(math.log2(i+1+1))
    return sum

def calculate_ndcgs(queries_dict, query_to_corpus_dict, index_url, search_field, fts_size, ndcgs_size, rerank_fun):
    
    ndcgs = []

    for q_id, q_text in queries_dict.items():
        ideal_search = list(query_to_corpus_dict[q_id].keys())[:ndcgs_size]
        idcg = calculate_dcg(ideal_search, query_to_corpus_dict[q_id])

        real_search_with_scores = find_for_phrase_with_exclusion(index_url, q_text, search_field, fts_size, [])
        reranked_search_with_scores = rerank_fun(q_text, real_search_with_scores)
        
        dcg = calculate_dcg(list(reranked_search_with_scores.keys())[:ndcgs_size], query_to_corpus_dict[q_id])

        ndcgs.append(dcg/idcg)

    return ndcgs