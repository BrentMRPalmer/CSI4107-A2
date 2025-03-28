import time
import csv
import sys

test_csv_path = sys.argv[1] if len(sys.argv) >= 2 else "./scifact/qrels/test.tsv"
formatted_csv_path = sys.argv[2] if len(sys.argv) >= 3 else "./scifact/qrels/formatted_test.tsv"

with open(test_csv_path, 'r', encoding="utf-8") as original_file:
    with open(formatted_csv_path, "w", newline='') as formatted_file:
        tsv_reader = csv.reader(original_file, delimiter="\t")
        tsv_writer = csv.writer(formatted_file, delimiter='\t')
        next(tsv_reader, None)
        for query in tsv_reader:
            query.insert(1, 0)
            if int(query[0]) % 2 == 1:
                tsv_writer.writerow(query)