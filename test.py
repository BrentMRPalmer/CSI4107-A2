bm25_scores = {"doc1": 0.524, "doc2": 0.202, "doc3": 0.297}

sorted_bm25 = dict(sorted(bm25_scores.items(), key=lambda item: item[1]))
print(sorted_bm25)