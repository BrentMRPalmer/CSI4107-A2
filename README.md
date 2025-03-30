# Assignment 2: Neural Information Retrieval System

## Table of Contents

- [Names and Student Numbers](#names-and-student-numbers)
- [Distribution of Work](#distribution-of-work)
- [Summary of Results](#summary-of-results)
- [Functionality of Programs](#functionality-of-programs)
- [How to Run](#how-to-run)
- [Description of Algorithms, Data Structures, and Optimizations](#description-of-algorithms-data-structures-and-optimizations)
- [Discussion of Results](#discussion-of-results)
- [References](#references)

## Names and Student Numbers

Brent Palmer (300193610) <br>
Jay Ghosh (300243766) <br>
Natasa Bolic (300241734)

## Distribution of Work

Brent Palmer

- Tested and documented results for the 36 sentence transformer models
- Code for document reranking using coco model
- Report

Jay Ghosh

- Searching for effective models (coco, oai, etc.)
- Code for document reranking using coco model
- Report

Natasa Bolic

- Code for document reranking using sentence transformer
- Code for document reranking using coco model
- Trying new models (sentence transformer, doc2vec, etc.)
- Report

## Summary of Results

We were able to improve our retrieval and ranking pipeline from Assignment 1 using neural based methods (`cocodr-large-msmarco`), resulting in a **MAP score to 0.6580** and **P@10 of 0.0927**.

## Functionality of Programs

A detailed description of the functionality of our program is included below, divided by step of the project.

This section emphasizes a high-level description of the responsibilities, including the inputs and outputs,
of each of the main stages in the programs. A discussion of the algorithms, data structures, and optimizations will be
included in [Description of Algorithms, Data Structures, and Optimizations](#description-of-algorithms-data-structures-and-optimizations).

The code is split into three files.

- **golden_retriever_sentence_transformer.py**: Retrieves documents using BM25, then reranks the top 100 documents using sentence transformers (36 possible models).
- **golden_retriever_coco.py**: Retrieves documents using BM25, then reranks the top 100 documents using `cocodr-large-msmarco`.
- **trec_processor.py**: An additional file that is used to clean the Scifact `test.tsv` file.

### Golden Retriever

**We now have two golden retriever files—both variants of Golden Retriver with specialized neural implementations. Both use the same base retrieval and ranking pipeline using BM25.**

The main python files that contain the entire information retrieval system, including preprocessing, indexing, retrieval and ranking, and re-ranking are the two variants of `golden_retriever.py`. They also read in the Scifact dataset corpus and queries, and output the ranked results in `results/Results_model_name.txt` where `model_name` is the name of the model. For the purposes of this description, any mention of `golden_retriever.py` refers to both `golden_retriever_sentence_transformer.py` and `golden_retriever_coco.py`.

#### Entry Point (Main)

The `golden_retriever.py` file begins in main, where it first reads the arguments representing the file paths
of the corpus and the query. It then calls `load_and_rank` method, which is the pipeline that preprocesses, indexes, retrieves
and ranks, and re-ranks (using neural methods) the documents.

Within `load_and_rank`, first the queries and documents are read in and stored. Each query and document is preprocessed as described below in
[Step 1: Preprocessing](#step-1-preprocessing) as it is read in. We keep an unprocessed (just lowercase query and document text) version of both the queries and the documents to be used for the neural network. Note that the unprocessed version is required for the neural network to capture semantic information. For example, removing stopwords could change the semantic meaning of a sentence. Then, the inverted index is created as described below
in [Step 2: Indexing](#step-2-indexing). Afterwards, a matrix containing all of the BM25 scores for each query and document is
computed. Next, the documents are ranked for each of the queries. The matrix formation process and ranking is described in
[Step 3: Retrieval and Ranking](#step-3-retrieval-and-ranking). Finally, the top 100 results are re-ranked using neural methods, which is also described in Step 3. The results are saved in `results/Results_model_name.txt` where `model_name` is the name of the model.

#### Step 1: Preprocessing

The preprocessing pipeline requires some global data that should only be initialized once, so it is initialized
at the start of the `golden_retriever.py` file. This includes the set of stopwords to be removed, as well as
the NLTK stemmer.

The preprocessing pipeline handles the preprocessing for both queries and documents. For both queries and documents, a function that orchestrates
the preprocessing pipeline is called. This method calls four different methods each with
distinct responsibilities:

- First, the raw JSON-formatted query or document is passed to a function that extracts the desired text and returns it in lowercase
  as a string.
- Second, the lowercase text is passed to a function that tokenizes it (converts it into a list of the words and punctuation).
- Third, the list of tokens is passed to a function that removes all stopwords, returning the filtered list.
- Finally, the filtered list of tokens is passed to a stemmer, that stems each token in the list.

Altogether, the overall pipeline function returns the extracted lowercase, tokenized, filtered, and stemmed list of tokens.

#### Step 2: Indexing

After the `load_and_rank` function calls the preprocessing pipeline as step 1, step 2 involves creating an inverted index using
the preprocessed document corpus. A single function is used to create the inverted index, taking the preprocessed representation of the document corpus
as input. Iterating over all of the documents, the counts of each token are calculated and inserted into the inverted index. The constructed
inverted index is returned.

#### Step 3: Retrieval and Ranking

After the `load_and_rank` function calls the indexing function as step 2, step 3 involves calculating the BM25 scores and ranking the documents.
Firstly, a function takes the preprocessed document corpus, the inverted index, and the calculated average document length as input to
compute a matrix that stores the BM25 score of every combination of term in our vocabulary with every document. To do so, each combination
of term and document is sent to a function that calculates the BM25 score of the provided combination. Finally, the computed BM25 matrix is
used to rank the documents. For querying, a single method is called which takes the preprocessed query, the preprocessed document corpus,
the computed BM25 matrix, and the inverted_index as input, and computes the ranking. The score of a document is the sum of the BM25 scores
between each term in the query and the document. The function produces a list of tuples in the form (document_id, BM25_score), sorted by
BM25_score in descending order. We then use a neural model to re-rank the top 100 documents for each query by generating the corresponding query and document embeddings, utilizing cosine similarity to determine relevance. The final ranked output is the re-ranked list of document IDs sorted by cosine similarity in descending order. The top 100 results for each query are then saved to `results/Results_model_name.txt` where `model_name` is the name of the model. The results stored in `results/Results_model_name.txt` can then be passed to the `trec_eval` script alongside the `formatted_test.tsv` file to compute the MAP and P@10 scores.

### Trec Processor (Cleaning trec.tsv)

There is a second program that cleans the provided `test.tsv` file. The original file is missing a column of zeroes, so it does not work
with the `trec_eval` script. Thus, the `trec_processor.py` file is used to process the `test.tsv` file, formatting it correctly such that it can be used with the `trec_eval` script. Having been passed the `test.tsv` file path as input and a target file path as output, a `formatted_test.tsv` file is generated which can be passed to `trec_eval`.

## How to Run

### Dependencies

Install dependencies by running `pip install -r requirements.txt` in the root directory of the project.

The Scifact dataset is available [here](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip).

### Execution Instructions

- Generate `results/Results_model_name.txt`, which contains the ranking results, by running `golden_retriever_sentence_transformer.py` or `golden_retriever_coco.py` with command line arguments. Please note that `golden_retriever_sentence_transformer.py` defaults to using the best model we found, `all-mpnet-base-v1`. Multiple models can be computed at once by changing the initialization of `models` on line 411. All models can be run by using the provided list on line 371.
  - Command Line Argument 1: The file path to the directory storing the `corpus.jsonl` and `queries.jsonl` files (default is `./scifact/`)
  - Command Line Argument 2: The name of the corpus json file (default `corpus.jsonl`)
  - Command Line Argument 3: The name of the query json file (default is `queries.jsonl`)
  - Example: `python golden_retriever.py ./scifact/ corpus.jsonl queries.jsonl`
- Clean the `Scifact` `qrels` file `test.tsv` by running `trec_processor.py` with command line arguments.
  - Command Line Argument 1: The file path of the original `test.tsv` file (default is `./scifact/qrels/test.tsv`)
  - Command Line Argument 2: The target file path of the `formatted_test.tsv` file that will be generated (default is `./scifact/qrels/formatted_test.tsv`)
    with the `trec_eval` script.
  - Example: `python trec_processor.py ./scifact/qrels/test.tsv ./scifact/qrels/formatted_test.tsv`
- Evaluate the `Results.txt` against the `formatted_test.tsv` file using `trec_eval` script.
  - In a Unix environment, download, extract, and build the `trec_eval` script using the following commands:
  <pre>
  wget https://trec.nist.gov/trec_eval/trec_eval_latest.tar.gz
  tar -xvzf trec_eval_latest.tar.gz
  cd trec_eval-9.0.7/
  make</pre>
  - Once the `trec_eval` script is built, it can be used with the following command line arguments:
    - Command Line Argument 1: The file path of the `Results.txt` file
    - Command Line Argument 2: The file path of the `formatted_test.tsv` file
    - Example: `./trec_eval -m map /mnt/c/Season11/CSI4107/CSI4107-A1/scifact/qrels/formatted_test.tsv /mnt/c/Season11/CSI4107/CSI4107-A1/Results.txt`

## Description of Algorithms, Data Structures, and Optimizations

### Golden Retriever

**We now have two files—both variants of Golden Retriver with specialized neural implementations. Both use the same base retrieval and ranking pipeline using BM25.**

#### Step 1: Preprocessing

This code begins by defining a global set of stopwords, a choice that provides constant-time membership checks (O(1) on average). The stopwords come from three sources: NLTK’s built-in English stopwords list, any custom words the user may specify, and a file of additional words (`provided_stopwords.txt`). By centralizing these words in a set, the code efficiently filters out common or uninformative terms during token processing, improving both performance and clarity in subsequent analyses.

To prepare raw text, the program relies on Python’s `json.loads` function, converting JSON-formatted strings into dictionaries. This step makes it straightforward to extract specific fields from each document or query, such as `title`, `text`, or `query` content. Once the relevant fields are identified, the code normalizes them by converting every character to lowercase. It then uses NLTK’s `word_tokenize` function to split the text into tokens, isolating each meaningful unit. After tokenization, the code applies a filtering procedure to remove any token found in the global `stop_words` set, as well as any token that lacks at least one alphabetical character. This filtration process ensures that numbers, punctuation, and other such elements do not pollute the final token list.

Finally, the code performs stemming using an NLTK `PorterStemmer` instance, reducing words to their base or root form. This transformation aims to unify variations of the same word—improving search or retrieval tasks where morphological differences should not matter. Dedicated functions, such as `preprocess_document_title_and_text`, `preprocess_document_title`, and `preprocess_query`, orchestrate this process. They each handle extraction of the relevant JSON fields (e.g., just `title` vs. `title` and `text`) and then pass the text through the lowercase, tokenize, remove-stopwords, and stem pipeline.

#### Step 2: Indexing

This code builds an inverted index by iterating through a corpus of preprocessed documents, where each document is indexed by its ID. For each document, the algorithm retrieves two things: the total number of tokens and the list of stemmed, stopword-filtered tokens. It then uses Python’s `Counter` class to count how many times each token appears within that document, allowing the index construction to also record term frequencies.

The resulting index is a dictionary in which each token maps to a list of two elements. The first element in this list is an integer denoting how many distinct documents contain that token, i.e. the document frequency. The second element is itself a list of tuples, where each tuple contains a document ID and the count of how many times that token appeared in that document.

#### Step 3: Retrieval and Ranking

The retrieval phase begins with `compute_bm25`, a function that calculates the BM25 score between a single term and a document. Inside `compute_bm25`, the algorithm first determines the term frequency (tf) for the chosen document by looking up the document’s token count in the inverted index. It then applies the classic BM25 formula (referencing Week 2 from the course website) which takes into account the document frequency. The adjustable hyperparameters `k1` and `b` control term frequency saturation and length normalization, respectively. We chose the values for them based on trial and error.

Next, `compute_bm25_matrix` computes BM25 scores for every term in the vocabulary against every document. It outputs a nested dictionary (the BM25 matrix) where the top-level keys are terms, and each term maps to a dictionary keyed by document ID: BM25 score pairs. This precomputation makes subsequent queries faster, since lookups require only a direct index access rather than repeated BM25 calculations.

The `rank` function then aggregates BM25 scores for a given query. It starts with a default score of zero for all documents, iterates over each query term, and sums that term’s BM25 score for any document that contains it. Documents that do not include the term are not updated. The function produces a sorted list of `(document_id, BM25_score)` tuples in descending order of their BM25 scores.

The top 100 results for each query are re-ranked using neural methods. We generate embeddings for both the query and its top 100 results, utilizing a document embedding cache named `cached_embeddings` to improve performance. Our rationale is that throughout the retrieval and ranking process—across all queries—certain documents are likely to appear multiple times. In these cases, recomputing embeddings from scratch would be inefficient. The `cached_embeddings` cache is implemented using a dictionary. Note that during this re-ranking process, we use the unprocessed document contents to generate the document embeddings. This is done to ensure all semantics of the documents are preserved and captured by the model.

#### Top 100 Results

`load_and_rank` orchestrates the entire pipeline by reading and preprocessing the document corpus, constructing the inverted index, computing the BM25 matrix, processing each query by calling the `rank` function, and then re-ranking the output using neural methods. It takes the top 100 re-ranked documents for each processed query and writes the output to a file. The output records the query ID, document ID, rank position, cosine similarity score, and a tag indicating whether the ranking included “text_included” or “title_only” data.

#### Optimizations in Model Inference

We ran our model using PyTorch on an RTX 4080 GPU, which provided substantial compute for both embedding generation and reranking. To optimize performance and ensure consistent outputs during inference, we set the model to evaluation mode using `model.eval()`—this disabled dropout and other training-specific layers. We also wrapped inference code in `torch.no_grad()` blocks to avoid unnecessary gradient tracking and reduce memory usage.

### Trec Processor (Cleaning trec.tsv)

A tab-separated test set is read from a file while the header row is skipped. For each subsequent row, a zero is inserted into the second column.

## Discussion of Results

### First 10 answers to queries 1 and 3

#### Best Model: `cocodr-large-msmarco`

##### Query 1

Below are the first 10 results of our best neural information retrieval system (`cocodr-large-msmarco`) for the query with ID `1`. These were accessed from the `Results.txt` file, which contains the top 100 documents for every query.

Query 1 is `0-dimensional biomaterials show inductive properties.`

```
1 Q0 17388232 1 0.927355170249939 cocodr-large-msmarco
1 Q0 35008773 2 0.9252241849899292 cocodr-large-msmarco
1 Q0 43385013 3 0.918635904788971 cocodr-large-msmarco
1 Q0 37437064 4 0.9175436496734619 cocodr-large-msmarco
1 Q0 10786948 5 0.9169590473175049 cocodr-large-msmarco
1 Q0 7581911 6 0.9167959094047546 cocodr-large-msmarco
1 Q0 1499964 7 0.9163487553596497 cocodr-large-msmarco
1 Q0 16541762 8 0.9160013198852539 cocodr-large-msmarco
1 Q0 21456232 9 0.915531575679779 cocodr-large-msmarco
1 Q0 1855679 10 0.9155093431472778 cocodr-large-msmarco
```

For the first query, the similarity scores of the top 10 documents are all quite high, ranging from `0.9155` to `0.9274`. From looking at the content of the highest ranked document, document `17388232` titled "Mechanical regulation of cell function with geometrically modulated elastomeric substrates," it appears to discuss the impact of elastomeric substrates on cells, specifically on their mechanical regulation. From our understanding, the query is about the impact of 0-dimensional biomaterials on cells, specifically on their inductive properties. Although these are not exactly the same topic, the query and the document both discuss interactions between cells and materials. This demonstrates the power of using models that can capture semantic meaning because despite the lack of matching keywords, they are identified as similar.

##### Query 3

Below are the first 10 results of our best neural information retrieval system (`cocodr-large-msmarco`) for the query with ID `3`. These were accessed from the `Results.txt` file, which contains the top 100 documents for every query.

Query 3 is `1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.`

```
3 Q0 2739854 1 0.9574798345565796 cocodr-large-msmarco
3 Q0 14717500 2 0.9471142888069153 cocodr-large-msmarco
3 Q0 23389795 3 0.944831371307373 cocodr-large-msmarco
3 Q0 32181055 4 0.9438236951828003 cocodr-large-msmarco
3 Q0 19058822 5 0.940618097782135 cocodr-large-msmarco
3 Q0 1388704 6 0.9366694092750549 cocodr-large-msmarco
3 Q0 9196472 7 0.936307966709137 cocodr-large-msmarco
3 Q0 1836154 8 0.9348548650741577 cocodr-large-msmarco
3 Q0 10944947 9 0.9325252175331116 cocodr-large-msmarco
3 Q0 41782935 10 0.9325160980224609 cocodr-large-msmarco
```

For the third query, the similarity scores are even higher than the first query for the top 10 documents, ranging from approximately `0.9325` to `0.9575`. From looking at the content of the highest ranked document, document `2739854` titled "Rare and common variants: twenty arguments," we observe that both the query and the document focus on rare versus common variants in genomes, noting that the effect of rare variants is larger than common variants. The topics appear to be very similar, which explains why the similarity score for the top document of this query is higher than the similarity score of the top document of the first query.

#### Second Best Model: `all-mpnet-base-v1`

##### Query 1

Below are the first 10 results of our second best neural information retrieval system (`all-mpnet-base-v1`) for the query with ID `1`. These were accessed from the `Results_all-mpnet-base-v1.txt` file, which contains the top 100 documents for every query.

Query 1 is `0-dimensional biomaterials show inductive properties.`

```
1 Q0 17388232 1 0.3453291356563568 all-mpnet-base-v1
1 Q0 21456232 2 0.26767271757125854 all-mpnet-base-v1
1 Q0 1065627 3 0.2575644254684448 all-mpnet-base-v1
1 Q0 19651306 4 0.24826312065124512 all-mpnet-base-v1
1 Q0 12824568 5 0.2216578722000122 all-mpnet-base-v1
1 Q0 17123657 6 0.20931318402290344 all-mpnet-base-v1
1 Q0 10931595 7 0.20020157098770142 all-mpnet-base-v1
1 Q0 4435369 8 0.15860764682292938 all-mpnet-base-v1
1 Q0 10786948 9 0.15089626610279083 all-mpnet-base-v1
1 Q0 21257564 10 0.14642101526260376 all-mpnet-base-v1
```

For the first query, the similarity scores are all quite low, ranging from approximately `0.1464` to `0.3453`. The highest ranked document, document `17388232` titled "Mechanical regulation of cell function with geometrically modulated elastomeric substrates," is the same as the highest ranked document from the best model. However, the score here is only `0.3453`, while the score from the best model is `0.9274`. The document that this model ranks as second highest, document `21456232` titled "A graphene-based platform for induced pluripotent stem cells culture and differentiation," is ranked 9th using the best model. This document also discusses the impact of a material (graphene) on cell behaviour, suggesting that it is a similar topic. This time, however, the document discusses the impact on a specific type of cell (induced pluripotent stem cells), showing that perhaps the document is slightly less relevant to the query.

##### Query 3

Below are the first 10 results of our second best neural information retrieval system (`all-mpnet-base-v1`) for the query with ID `3`. These were accessed from the `Results_all-mpnet-base-v1.txt` file, which contains the top 100 documents for every query.

Query 3 is `1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.`

```
3 Q0 1388704 1 0.6403359174728394 all-mpnet-base-v1
3 Q0 2739854 2 0.621632993221283 all-mpnet-base-v1
3 Q0 23389795 3 0.601943850517273 all-mpnet-base-v1
3 Q0 19058822 4 0.5938122868537903 all-mpnet-base-v1
3 Q0 1544804 5 0.5918434262275696 all-mpnet-base-v1
3 Q0 3662132 6 0.5870236158370972 all-mpnet-base-v1
3 Q0 15153602 7 0.5730934143066406 all-mpnet-base-v1
3 Q0 14717500 8 0.57110995054245 all-mpnet-base-v1
3 Q0 13914198 9 0.5694984197616577 all-mpnet-base-v1
3 Q0 10145528 10 0.5681357383728027 all-mpnet-base-v1
```

Once again, the similarity scores are higher for query 3, ranging from approximately `0.5681` to `0.6403`. The highest ranking document, document `1388704` titled "The essence of SNPs," was ranked 6th using the best model. Both the document and the query discuss genome variation, however, the document focuses on common variants (SNPs) while the query focuses on rare variants. This suggests that, although relevant, this document may not be as relevant as the document ranked first by the best model. However, the second highest ranking document, document `2739854` titled "Rare and common variants: twenty arguments," is actually the highest ranked document using the best model. These similar rankings between the two models are expected since the MAP scores of the two models are fairly similar and their P@10 scores are identical.

### Evaluation Results (MAP and P@10 Score)

We provide two tables summarizing the performance of our neural information retrieval systems, reporting both MAP and P@10. The first table shows the results of the top 2 models (`cocodr-large-msmarco` and `all-mpnet-base-v1`), while the second table covers all 37 tested models.

#### Top 2 Model Results

Our top 2 models are `cocodr-large-msmarco` and `all-mpnet-base-v1`. Their results are highlighted below.

| Model Name           | MAP Score | P@10   |
| -------------------- | --------- | ------ |
| cocodr-large-msmarco | 0.6580    | 0.0927 |
| all-mpnet-base-v1    | 0.6289    | 0.0927 |

#### All Tested Model Results

*We tested 36 sentence transformers and cocodr-large-msmarco.*

| Model Name                            | MAP Score | P@10   |
| ------------------------------------- | --------- | ------ |
| msmarco-bert-base-dot-v5              | 0.5508    | 0.0830 |
| multi-qa-MiniLM-L6-dot-v1             | 0.5316    | 0.0803 |
| sentence-t5-base                      | 0.4491    | 0.0753 |
| msmarco-distilbert-base-tas-b         | 0.5348    | 0.0810 |
| msmarco-distilbert-dot-v5             | 0.5025    | 0.0783 |
| paraphrase-distilroberta-base-v2      | 0.5791    | 0.0850 |
| paraphrase-MiniLM-L12-v2              | 0.5661    | 0.0847 |
| paraphrase-multilingual-mpnet-base-v2 | 0.4883    | 0.0790 |
| paraphrase-TinyBERT-L6-v2             | 0.5478    | 0.0877 |
| paraphrase-MiniLM-L6-v2               | 0.5156    | 0.0803 |
| paraphrase-albert-small-v2            | 0.5227    | 0.0810 |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.4765    | 0.0770 |
| paraphrase-MiniLM-L3-v2               | 0.4987    | 0.0777 |
| distiluse-base-multilingual-cased-v1  | 0.4516    | 0.0727 |
| distiluse-base-multilingual-cased-v2  | 0.4102    | 0.0697 |
| average_word_embeddings_komninos      | 0.3136    | 0.0557 |
| average_word_embeddings_glove.6B.300d | 0.3059    | 0.0497 |
| gtr-t5-large                          | 0.6241    | 0.0870 |
| all-mpnet-base-v1                     | 0.6289    | 0.0927 |
| multi-qa-mpnet-base-dot-v1            | 0.5771    | 0.0853 |
| multi-qa-mpnet-base-cos-v1            | 0.5763    | 0.0880 |
| all-roberta-large-v1                  | 0.6179    | 0.0897 |
| sentence-t5-xl                        | 0.5097    | 0.0823 |
| all-distilroberta-v1                  | 0.5985    | 0.0890 |
| all-MiniLM-L12-v1                     | 0.6245    | 0.0907 |
| all-MiniLM-L12-v2                     | 0.5968    | 0.0897 |
| multi-qa-distilbert-dot-v1            | 0.5853    | 0.0847 |
| multi-qa-distilbert-cos-v1            | 0.5797    | 0.0840 |
| gtr-t5-base                           | 0.5834    | 0.0863 |
| sentence-t5-large                     | 0.5110    | 0.0800 |
| all-MiniLM-L6-v2                      | 0.6190    | 0.0903 |
| multi-qa-MiniLM-L6-cos-v1             | 0.5381    | 0.0840 |
| all-MiniLM-L6-v1                      | 0.5889    | 0.0893 |
| paraphrase-mpnet-base-v2              | 0.5637    | 0.0880 |
| all-mpnet-base-v2                     | 0.6254    | 0.0927 |
| gtr-t5-xl                             | 0.6247    | 0.0893 |
| cocodr-large-msmarco                  | 0.6580    | 0.0927 |

> **MAP** = Mean Average Precision  
> **P@10** = Precision at 10

Transformer models encode sections of text into vectors that capture semantic meaning. We tested 36 different transformer models from the `Sentence Transformers` library, obtaining MAP scores in the range of `0.3059` to `0.6289`.

The transformer from the `Sentence Transformers` library that performed the best with respect to MAP was `all-mpnet-base-v1`, acheiving a **MAP score of 0.6289** and a **P@10 of 0.0927**. This model encodes the documents into 768-dimensional dense vectors, allowing it to capture a high level of information. The model makes use of the pretrained `microsoft/mpnet-base model`, then fine-tunes it using a dataset consisting of 1 billion sentence pairs. The fine-tuning involves trying to predict the corresponding sentence given one of the sentences from a pair and refining based on the cross entropy loss.

Despite accounting for semantic meaning, the model does not acheive a higher MAP score than in Assignment 1 (`0.6310`), and does not achieve a higher precision score than in Assignment 1 (`0.0948`).

The model trains on text with a maximum word length of 128, while the average number of words of a document in our corpus is approximately 219 words. This discrepancy could explain the reduction in performance, since the reranking is optimized for shorter documents than what we provide.

Overall, sentence transformers do not improve the performance of our system, leading us to look into other types of re-ranking models.

The **best-performing model** for our submission was `OpenMatch/cocodr-large-msmarco`, achieving a **MAP score of 0.6580** and a **P@10 of 0.0927**, which is a higher MAP score than in Assignment 1 (`0.6310`), but slightly lower precision score than in Assignment 1 (`0.0948`). The model is based on the BERT-large architecture, comprising 24 transformer layers with a hidden size of 1024, totaling approximately 335 million parameters. This deep architecture enables the model to capture intricate patterns and relationships within text data.

The model was pretrained on the BEIR corpus using Continuous Contrastive Learning (COCO). This method involves treating sequences from the same document as positive pairs and sequences from different documents as negative pairs, enhancing the model's ability to discern subtle semantic differences. Subsequently, the model was fine-tuned on the MS MARCO dataset employing implicit Distributionally Robust Optimization (iDRO). This technique dynamically adjusts the training focus on different query clusters, ensuring the model remains robust across various data distributions and performs well even on underrepresented query types.

## References

Stopword removal (of english words): https://www.geeksforgeeks.org/removing-stop-words-nltk-python/<br>
command line args https://www.geeksforgeeks.org/command-line-arguments-in-python/<br>
Remove words without letters (numbers, symbols, etc.): https://www.w3schools.com/python/ref_string_isalpha.asp + https://www.w3schools.com/python/ref_func_any.asp<br>
Porter stemming: https://www.geeksforgeeks.org/python-stemming-words-with-nltk/<br>
Sorted dictionary: https://www.datacamp.com/tutorial/sort-a-dictionary-by-value-python<br>
Writing to tsv: https://medium.com/@nutanbhogendrasharma/creating-and-writing-to-different-type-of-files-in-python-6a2a1579bc25<br>
Read tsv: https://www.geeksforgeeks.org/simple-ways-to-read-tsv-files-in-python/<br>
Best sentence transformer model: https://huggingface.co/sentence-transformers/all-mpnet-base-v1 <br>
Cuda: https://pytorch.org/docs/stable/cuda.html <br>
psutil: https://psutil.readthedocs.io/en/latest/#processes <br>
Process Priority: https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setpriorityclass <br>
system: https://docs.python.org/3/library/sys.html <br>
tqdm: https://pypi.org/project/tqdm/#iterable-based <br>
OpenMatch Coco usage: https://github.com/OpenMatch/COCO-DR?tab=readme-ov-file <br>
Coco Hugging Face: https://huggingface.co/OpenMatch/cocodr-large-msmarco <br>
Coco research paper: https://arxiv.org/abs/2210.15212 <br>
Tokenizer: https://huggingface.co/docs/transformers/en/fast_tokenizers <br>
Model eval: https://www.geeksforgeeks.org/what-does-model-eval-do-in-pytorch/ <br>
Remove training mechanisms: https://pytorch.org/docs/stable/generated/torch.no_grad.html <br>
Sentence transformers list: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html <br>
Sentence transformers usage: https://huggingface.co/sentence-transformers <br>
Sentence transformer device, convert to tensor: https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html <br>
Model to CUDA: https://medium.com/%40transformergpt/how-to-convert-sentence-transformer-pytorch-models-to-onnx-with-the-right-pooling-method-61b1c83515d2 <br>
Cosine similarity: https://pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html <br>
