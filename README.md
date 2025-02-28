## Natural Language Processing

This repository offers solutions to a diverse set of NLP exercises, each exploring a different topic. The labs range from regular expressions and full-text search with Elasticsearch to advanced question-answering using RAG and neural models.

### Tabular content description
All the exercises have been solved from scratch (without a provided template notebook), and the most interesting ones are presented in bold.

|        | Topic                                             | Instruction                 | Solution                     | Libs*/Tools                        | Points    |
|--------|---------------------------------------------------|-----------------------------|------------------------------|------------------------------------|-----------|
| 1      | Regular expressions                               | 1-regexp.md                 | regex-ex.ipynb               |                                    | 10/10     |
| 2      | Lemmatization and full text search (FTS)          | 2-fts.md                    | fts.ipynb                    |                                    | 10/10     |
| **3**  | **Levenshtein distance and spelling corrections** | **3-levenshtein.md**        | **levensthein.ipynb**        | **spacy**                          | **10/10** |
| 4      | Language modelling                                | 4-lm.md                     | lab4.ipynb                   | transformers                       | 10/10     |
| **5**  | **Text classification**                           | **5-classification.md**     | **lab5.ipynb**               | **torch, transformers**            | **10/10** |
| 6      | Named entity recognition                          | 6-ner.md                    | ner.ipynb                    | spacy                              | 10/10     |
| 7      | NER and Classification with LLMs                  | 7-classification-ner-llm.md | classification-ner-llm.ipynb | ollama                             | 9/10      |
| **8**  | **Neural search for question answering**          | **8-neural.md**             | **neural.ipynb**             | **haystack, qdrant, transformers** | **10/10** |
| **9**  | **Contextual question answering**                 | **9-qa.md**                 | **qa.ipynb**                 | **spacy, transformers**            | **10/10** |
| **10** | **RAG-based Question Answering**                  | **10-rag.md**               | **rag.ipynb**                | **langchain, ollama, qdrant**      | **10/10** |

*Only the most relevant libraries are listed; basic ones like `regex`, `sklearn` or `datasets` are not included in the table.