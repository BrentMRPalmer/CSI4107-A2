# Assignment 1: Information Retrieval System

## Table of Contents
- [Names and Student Numbers](#names-and-student-numbers)
- [Distribution of Work](#distribution-of-work)
- [Summary of Results](#summary-of-results)
- [Functionality of Programs](#functionality-of-programs)
- [How to Run](#how-to-run)
- [Description of Algorithms, Data Structures, and Optimizations](#description-of-algorithms-data-structures-and-optimizations)
- [Vocabulary](#vocabulary)
- [Mean Average Precision (MAP) Score](#mean-average-precision-map-score)

## Names and Student Numbers
Brent Palmer (300193610) <br>
Jay Ghosh (300243766) <br>
Natasa Bolic (300241734)

## Distribution of Work
Brent Palmer
- Step 1: Preprocessing
- Step 2: Indexing
- Step 3: Retrieval and Ranking
- Cleaning test.tsv
- Refactoring and Optimizations
- Docstrings
- Report

Jay Ghosh
- Refactoring and Optimizations
- Docstrings
- Report

Natasa Bolic
- Step 1: Preprocessing
- Step 2: Indexing
- Step 3: Retrieval and Ranking
- Cleaning test.tsv
- Top 100 results and results.txt
- Running trec_eval

## Summary of Results

We achieved a mean average precision score of 0.6310 using the BM25 approach.

## Functionality of Programs

A detailed description of the functionality of our program is included below, divided by step of the project.

This section emphasizes a high-level description of the responsibilities, including the inputs and outputs,
of each of the main stages in the programs. A discussion of the algorithms, data structures, and optimizations will be
included in [Description of Algorithms, Data Structures, and Optimizations](#description-of-algorithms-data-structures-and-optimizations).

The code is split into two files.
- **golden_retriever.py**: The main python file that contains the entire information retrieval system, including preprocessing, indexing,
and retrieval and ranking.
- **trec_processor.py**: An additional file that is used to clean the Scifact `test.tsv` file.

### Golden Retriever

The main python file that contains the entire information retrieval system, including preprocessing, indexing,
and retrieval and ranking is called `golden_retriever.py`. It also reads in the Scifact dataset corpus and queries, 
and outputs the ranked results in `Results.txt`.

#### Entry Point (Main)

The `golden_retriever.py` file beings in main, where it first reads the arguments representing the file paths
of the corpus and the query. It then reads in, preprocesses (as described below in
[Step 1: Preprocessing](#step-1-preprocessing)), and stores all of the queries. The queries are passed to the 
`load_and_rank` method, which is the pipeline that preprocesses, indexes, and retrieves and ranks the documents.

The `load_and_rank` is called twice, once to rank the documents using the title and text of the documents (results saved 
in `Results.txt`), and again to rank the documents using only the titles (results saved in `Results_Title_Only.txt`).

Within `load_and_rank`, first the documents are read in and stored. Each document is preprocessed as described below in
[Step 1: Preprocessing](#step-1-preprocessing) as it is read in. Then, the inverted index is created as described below
in [Step 2: Indexing](#step-2-indexing). Afterwards, a matrix containing all of the BM25 scores for each query and document is
computed. Next, the documents are ranked for each of the queries. The matrix formation process and ranking is described in
[Step 3: Retrieval and Ranking](#step-3-retrieval-and-ranking). The results are saved in the `Results.txt` or `Results_Title_Only.txt`

#### Step 1: Preprocessing

The preprocessing pipeline requires some global data that should only be initialized once, so it is initialized 
at the start of the `golden_retriever.py` file. This includes the set of stopwords to be removed, as well as 
the NLTK stemmer.

The preprocessing pipeline handles the preprocessing for both queries and documents. The queries call the pipeline from
main, whereas the documents call this function as the first step of `load_and_rank`. For both queries and documents, a function that orchestrates 
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
used to rank the documents. For each odd-numbered query (note that the assignment description specifies that we must only calculate the scores
for odd-numbdered queries), a single method is called which takes the preprocessed query, the preprocessed document corpus,
the computed BM25 matrix, and the inverted_index as input, and computes the ranking. The score of a document is the sum of the BM25 scores 
between each term in the query and the document. The function returns a list of tuples in the form (document_id, BM25_score), sorted by 
BM25_score in descending order. This is the final ranking of all odd-numbered queries.

#### Top 100 Results

Despite having calculated the final document ranking, the assignment requires one more step: filtering the top 100 documents for each query,
and saving them to a file called `Results.txt`. Since the documents are ranked in sorted order, we simply loop over the first 100 documents
in the ranked list of tuples, and save their values to `Results.txt`. The results stored in `Results.txt` can then be passed to the
`trec_eval` script alongside the `formatted_test.tsv` file to compute the MAP score.

### Trec Processor (Cleaning trec.tsv)

There is a second program that cleans the provided `test.tsv` file. The original file is missing a column of zeroes, so it does not work
with the `trec_eval` script. Furthermore, it includes even-numbered queries. Thus, the `trec_processor.py` file is used to process 
the `test.tsv` file, formatting it correctly such that it can be used with the `trec_eval` script. Having been passed the `test.tsv` file 
path as input and a target file path as output, a `formatted_test.tsv` file is generated which can be passed to `trec_eval`.

## How to Run

### Dependencies

Install dependencies by running `pip install -r requirements.txt` in the root directory of the project.

The Scifact dataset is available [here](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip).

### Execution Instructions
- Generate `Results.txt`, which contains the ranking results, by running `golden_retriever.py` with command line arguments.
    - Command Line Argument 1: The file path to the directory storing the `corpus.jsonl` and `queries.jsonl` files (default is `./scifact/`)
    - Command Line Argument 2: The name of the corpus json file (default `corpus.jsonl`)
    - Command Line Argument 3: The name of the query json file (default is `queries.jsonl`)
    - Example: `python golden_retriever.py ./scifact/ corpus.jsonl queries.jsonl`
- Clean the `Scifact` `qrels` file `test.tsv` by running `trec_processor.py` with command line arguments.
    - Command Line Argument 1: The file path of the original `test.tsv` file (default is `./scifact/qrels/test.tsv`)
    - Command Line Argument 2: The target file path of the `formatted_test.tsv` file that will be generated (default is `./scifact/qrels/formatted_test.tsv`)
    - Note: This "cleaning" filters the test file to only odd queries, as per the assignment instructions, and adds a `0` column, for compatibility
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

The `rank` function then aggregates BM25 scores for a given query. It starts with a default score of zero for all documents, iterates over each query term, and sums that term’s BM25 score for any document that contains it. Documents that do not include the term are not updated. Finally, the function produces a sorted list of `(document_id, BM25_score)` tuples in descending order of their BM25 scores.

#### Top 100 Results

`load_and_rank` orchestrates the entire pipeline by reading and preprocessing the document corpus, constructing the inverted index, computing the BM25 matrix, and then processing each query by calling the `rank` function on those queries whose IDs are odd. It takes the top 100 ranked documents for each processed query and writes the output to a file. The output records the query ID, document ID, rank position, BM25 score, and a tag indicating whether the ranking included “text_included” or “title_only” data. 

### Trec Processor (Cleaning trec.tsv)

A tab-separated test set is read from a file while the header row is skipped. For each subsequent row, a zero is inserted into the second column, and the first column (representing the query ID) is converted to an integer to check if it is odd. Only rows with odd IDs are written to a new file. This process produces a reduced test set containing only the odd-numbered queries, formatted for TREC eval.

## Vocabulary

### How big was the vocabulary?

We ran `len(inverted_index)` to obtain the size of the vocabulary, and found that it contained 37975 terms.

### Sample of 100 tokens from the vocabulary
Below is a sample of 100 tokens from our vocabulary. To obtain these results, we ran a loop which iterates over the first 100 tokens in our inverted index and printed them.

```
microstructur
develop
human
newborn
cerebr
white
matter
assess
vivo
diffus
tensor
magnet
reson
imag
alter
architectur
brain
affect
cortic
result
function
disabl
line
scan
diffusion-weight
mri
sequenc
analysi
appli
measur
appar
coeffici
calcul
rel
anisotropi
delin
three-dimension
fiber
preterm
full-term
infant
effect
prematur
earli
gestat
studi
term
central
mean
wk
microm2/m
decreas
posterior
limb
intern
capsul
versu
closer
birth
absolut
valu
area
compar
nonmyelin
corpu
callosum
visibl
mark
differ
organ
data
indic
quantit
water
insight
live
induct
myelodysplasia
myeloid-deriv
suppressor
cell
myelodysplast
syndrom
md
age-depend
stem
malign
share
biolog
featur
activ
adapt
immun
respons
ineffect
hematopoiesi
report
mdsc
classic
link
```

### First 10 answers to the first 2 queries

#### Query 1
Below are the first 10 results of our information retrieval system for the first test query (note that the first odd query has ID 9). These were accessed from the `Results.txt` file, which contains the top 100 documents for every query.
```
9 Q0 44265107 1 18.958258391537722 text_included
9 Q0 24700152 2 7.423023189374037 text_included
9 Q0 25182647 3 6.6074423594848986 text_included
9 Q0 16737210 4 6.412449606330081 text_included
9 Q0 8190282 5 6.380871226577165 text_included
9 Q0 14647747 6 6.191347824712254 text_included
9 Q0 37699461 7 6.1785650232134355 text_included
9 Q0 21859699 8 6.100003218989789 text_included
9 Q0 28419824 9 5.822323687699198 text_included
9 Q0 26105746 10 5.529571232935707 text_included
```

#### Query 2
Below are the first 10 results of our information retrieval system for the second test query. These were accessed from the `Results.txt` file. These were accessed from the `Results.txt` file, which contains the top 100 documents for every query.
```
11 Q0 25510546 1 17.37933522482014 text_included
11 Q0 20904154 2 14.695012721017557 text_included
11 Q0 13780287 3 14.39400930168742 text_included
11 Q0 7482674 4 14.208599621494006 text_included
11 Q0 32587939 5 14.160733088926552 text_included
11 Q0 4399311 6 14.076685357651069 text_included
11 Q0 29459383 7 13.871039001428388 text_included
11 Q0 13958154 8 12.97664066012545 text_included
11 Q0 19708993 9 12.767749957207569 text_included
11 Q0 8453819 10 12.419113157787327 text_included
```

### Discussion of Results


## Mean Average Precision (MAP) Score

**Titles and Full Text:** 0.6310

**Titles Only Run:** 0.4023

### Discussion

We obtain better results when we use the titles and the full text, as opposed to simply the titles. More specifically, the mean average precision is 22.87% higher when we include the text of the documents as well. 

A possible reason is that the full text provides more context about the document. For example, the full text is more likely than just the title to include synonyms for important terms. So, if a query contains a synonym for a term in the title, it is possible that it would be matched with the document if the information retrieval system uses the full text, but not if it uses just the title.