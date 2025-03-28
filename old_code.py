# tests
# print(bm25("methyl", "3", documents, inverted_index, avdl))
# print(rank(queries["0"], documents, matrix))
# print(rank(["brain", "play", "disabl"], documents, matrix))

# print(matrix["play"]["3"])
# for doc_id, content in documents.items():
#     print(f"doc id: {doc_id} content: {content}")
#     time.sleep(10)

# for term, value in inverted_index.items():
#     print(f"term: {term} value: {value}")

# for document_id, _ in documents.items():
#     score = 0
#     for term in query:
#         if term in inverted_index:
#             score += matrix[term][document_id]
#     scores[document_id] = score

# return sorted(scores.items(), key=lambda item: item[1], reverse=True)