import math
from datasets import load_dataset

def load_fiqa():
    corpus = load_dataset("clarin-knext/fiqa-pl", name="corpus")
    queries = load_dataset("clarin-knext/fiqa-pl", name="queries")
    qrels = load_dataset("clarin-knext/fiqa-pl-qrels")
    return corpus, queries, qrels

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

def calculate_dcg(docs, docs_scoring):
    sum = 0
    for i, doc_id in enumerate(docs):
        if doc_id in docs_scoring.keys():
            sum += (2**docs_scoring[doc_id]-1)/(math.log2(i+1+1))
    return sum

def calculate_ndcgs(queries_dict, query_to_corpus_dict, query_pipeline, query_fun, ndcgs_size):
    
    ndcgs = []

    for q_id, q_text in queries_dict.items():
        ideal_search = list(query_to_corpus_dict[q_id].keys())[:ndcgs_size]
        idcg = calculate_dcg(ideal_search, query_to_corpus_dict[q_id])

        real_search_with_scores = query_fun(query_pipeline, q_text, ndcgs_size)

        search_result_docs = [int(real_search_doc.id) for real_search_doc in real_search_with_scores]
        
        dcg = calculate_dcg(search_result_docs, query_to_corpus_dict[q_id])

        ndcgs.append(dcg/idcg)

    return ndcgs

import matplotlib.pyplot as plt
import numpy as np

def present_results(ndcgs):

    zeros = []
    non_zeros = []

    for ndcg in ndcgs:
        if ndcg == 0:
            zeros.append(ndcg)
        else:
            non_zeros.append(ndcg)

    labels = ['>0 Results', '0 Results']
    sizes = [len(non_zeros), len(zeros)]

    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title('NDCG@5 Results Proportion: 0 to >0')
    plt.show()

    plt.hist(non_zeros, bins=10, edgecolor='black')
    plt.title("Histogram of Non-Zero NDCG@5")
    plt.xlabel("Score")
    plt.ylabel("Number of Documents")
    plt.show()

    print(f"NDCG@5 Mean: {np.mean(ndcgs)} and Std: {np.std(ndcgs)}")
    print(f"NDCG@5 > 0 Mean: {np.mean(non_zeros)} and Std: {np.std(non_zeros)}")
