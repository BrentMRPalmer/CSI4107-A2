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


## average word count of documents while reading in the documents (result: 218.63)
'''
    # Read in corpus
    with open(corpus_path, 'r', encoding="utf-8") as corpus:
        word_count = 0

        for document in corpus:
            # Load in the document in json format
            data = json.loads(document)

            # Compute length of document in words
            word_count += len(extract_document_title_and_text(document).split())

            if include_text:
                # Preprocess document title and text before assigning to dictionary
                documents[data["_id"]] = preprocess_document_title_and_text(document)
            else:
                # Preprocess document title before assigning to dictionary
                documents[data["_id"]] = preprocess_document_title(document)
        print(word_count/5183)
'''