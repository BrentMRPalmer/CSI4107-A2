import json
import math
import os
import psutil
import sys
import time
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

if sys.platform == "win32":
    p = psutil.Process()
    p.nice(psutil.REALTIME_PRIORITY_CLASS)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

#############
# Downloads
#############
nltk.download('punkt_tab')
nltk.download('stopwords')

####################
# Intialialization
####################

# Initialize set of stopwords

# Default set of English stopwords from nltk
stop_words = set(stopwords.words('english'))

# Any specific words we wanted to remove
custom_stopwords = {} 
stop_words.update(custom_stopwords)

# Stopwords recommended from the assignment description
with open("./resources/provided_stopwords.txt", 'r', encoding="utf-8") as provided_stopwords:
    for stopword in provided_stopwords:
        stop_words.add(stopword.strip())

# Initialize stemmer
stemmer = PorterStemmer()

##########################
# Step 1: Preprocessing
##########################

def extract_document(document):
    """ Extracts and concatenates the title and text from a JSON-formatted document string.
    
    Given a string representing the document in the original json format (with keys _id, title, text, and metadata),
    the function extracts the title and text. It concantenates them, putting an space in between. It also converts
    the string to lowercase.

    Args:
        document (str): A JSON-formatted string representing a document with keys _id, title, text, and metadata.
    
    Returns:
        str: A lowercase string containing the concatenated title and text.
    """
    data = json.loads(document)
    # Extract the title and text from document json, and make it lowercase
    if data['title'][-1] == '.':
        return f"{data['title']} {data['text']}".lower()
    # If the title does not end with a period, add one before combining
    else:
        return f"{data['title']}. {data['text']}".lower()

def extract_query_text(query):
    """ Extracts the query from a JSON-formatted query string.

    Given a string representing a query in the original json format (with keys _id, text, and metadata),
    the function extracts the text and converts it to lowercase.

    Args:
        query (str): A JSON-formatted string representing a query with keys _id, text, and metadata.
    
    Returns:
        str: A lowercase string containing the query text.
    """
    data = json.loads(query)
    # Extract the text from query json, and make it lowercase
    return f"{data['text']}".lower()

def remove_stopwords(tokens):
    """ Removes stopwords from a list of tokens.

    This function uses the set of stopwords defined globally in the Initialization section of the code.
    Any token from the set of stopwords is filtered out, as well as any token that does not have
    at least one English character. It can be used for both documents and queries.

    Args:
        tokens (list): The list of tokens to be filtered.

    Returns:
        list: The list of tokens with stopwords removed.
    """
    filtered_words = []
    for token in tokens:
        # Take out stopwords, special characters, and numbers
        # Removes anything without at least one character from the English alphabet
        if token not in stop_words and any(char.isalpha() for char in token):
            filtered_words.append(token)
    return filtered_words

def stem(tokens):
    """ Stems the list of tokens using NLTK's PorterStemmer.

    This function uses NLTK's PorterStemmer to stem each token in the provided
    list of tokens. It can be used for both documents and queries.

    Args:
        tokens (list): The list of tokens to be stemmed.

    Returns:
        list: The list of stemmed tokens.
    """
    stemmed_tokens = []
    for token in tokens:
        stemmed_tokens.append(stemmer.stem(token))
    return stemmed_tokens

def preprocess_document(document):
    """ Processes a JSON-formatted document string by setting all characters to lowercase,
    tokenizing, removing stopwords, and stemming. Extracts both title and text.

    Args:
        document (str): The JSON-formatted document string to be processed.

    Returns:
        tuple:
            int: Number of tokens.
            list: List of processed tokens.
    """
    text_document = extract_document(document)
    tokenized_document = word_tokenize(text_document) # Uses NLTK's function
    filtered_document = remove_stopwords(tokenized_document)
    stemmed = stem(filtered_document)
    return (len(stemmed), stemmed)

def preprocess_query(query):
    """ Processes a JSON-formatted query string by setting all characters to lowercase,
    tokenizing, removing stopwords, and stemming. Extracts text.

    Args:
        query (str): The JSON-formatted query string to be processed.

    Returns:
        list: List of processed tokens.
    """
    text_query = extract_query_text(query)
    tokenized_query = word_tokenize(text_query) # Uses NLTK's function
    filtered_query = remove_stopwords(tokenized_query)
    return stem(filtered_query)

###################
# Step 2: Indexing
###################

def create_inverted_index(documents):
    """ Creates an inverted index based on the provided preprocessed representation of the document corpus.

    Args:
        documents (dict): A dictionary mapping document ID to a tuple containing the number of tokens 
                            in the document and the corresponding list of tokens.
                            { document_id (int): ( number of tokens (int), list of tokens (list) ) (tuple) }

    Returns:
        dict: A dictionary mapping the token to a list that contains the number of documents with that token as the first element,
                and a list of tuples as the second element, where each tuple contains the document ID and the corresponding count of 
                the token in that document, for each document that contains that token.
                { token (str): [number of documents with token (int), [( document ID (int), token count (int) )]] (List[int, List[tuple]]) }
    """
    # Keys are tokens in vocabulary, values are pairs of document ID and token count
    inverted_index = dict()

    for document_id, content in documents.items():
        text = content[1]
        # Count the occurrences of each term in the current document's text
        counter = Counter(text)
        for token, count in counter.items():
            if token not in inverted_index:
                inverted_index[token] = [1, [(document_id, count)]]
            else:
                inverted_index[token][0] += 1
                inverted_index[token][1].append((document_id, count))

    return inverted_index

###############################
# Step 3: Retrieval and Ranking
###############################

def compute_bm25(term, document_id, documents, inverted_index, avg_dl):
    """ Computes the BM25 score between a given term and document.

    Uses the BM25 formula provided in the Week 2 section of the CSI4107 course page.

    Args:
        term (str): The term used to calculate BM25 score.
        document_id (int): The document ID used to calculate the BM25 score.
        documents (dict): A dictionary mapping document ID to a tuple containing the number of tokens 
                            in the document and the corresponding list of tokens.
                            { document_id (int): ( number of tokens (int), list of tokens (list) ) (tuple) }
        inverted_index (dict): A dictionary mapping the token to a list that contains the number of documents with that token as the first element,
                                and a list of tuples as the second element, where each tuple contains the document ID and the corresponding count of 
                                the token in that document, for each document that contains that token.
                                { token (str): [number of documents with token (int), [( document ID (int), token count (int) )]] (List[int, List[tuple]]) }
        avg_dl (float): The average length over documents.
    
    Returns:
        float: The BM25 score between the given term and document.
    """
    N = len(documents)
    tf = 0
    for current_id, count in inverted_index[term][1]:
        if current_id == document_id:
            tf = count
            break
    df = inverted_index[term][0]
    dl = documents[document_id][0]
    avdl = avg_dl
    k1 = 1.2
    b = 0.75
    return (tf * math.log((N - df + 0.5) / (df + 0.5))) / (k1 * ((1 - b) + (b * dl) / avdl) + tf)

def compute_bm25_matrix(documents, inverted_index, avg_dl):
    """ Generates a matrix containing the BM25 score of every combination of term in our vocabulary with every document.

    Args:
        documents (dict): A dictionary mapping document ID to a tuple containing the number of tokens 
                            in the document and the corresponding list of tokens.
                            { document_id (int): ( number of tokens (int), list of tokens (list) ) (tuple) }
        inverted_index (dict): A dictionary mapping the token to a list that contains the number of documents with that token as the first element,
                                and a list of tuples as the second element, where each tuple contains the document ID and the corresponding count of 
                                the token in that document, for each document that contains that token.
                                { token (str): [number of documents with token (int), [( document ID (int), token count (int) )]] (List[int, List[tuple]]) }
        avg_dl (float): The average length over documents.
    
    Returns:
        Dict[dict]: A matrix containing the BM25 score of every combination of term in our vocabulary with every document.
                        { term (str): {document_id (int): bm25 score (float)} }
    """
    matrix = dict()
    for term, _ in inverted_index.items():
        matrix[term] = dict()
        for document_id, _ in documents.items():
            matrix[term][document_id] = compute_bm25(term, document_id, documents, inverted_index, avg_dl)
    return matrix

def rank(query, documents, bm25_matrix, inverted_index):
    """ Retrieves and ranks documents in descending order based on summed BM25 scores.

    The score of a document is the sum of the BM25 scores between each term in the query and the document.

    Args:
        query (list): The list of query terms.
        documents (dict): A dictionary mapping document ID to a tuple containing the number of tokens 
                            in the document and the corresponding list of tokens.
                            { document_id (int): ( number of tokens (int), list of tokens (list) ) (tuple) }
        bm25_matrix (Dict[dict]): A matrix containing the BM25 score of every combination of term in our vocabulary with every document.
                                    { term (str): {document_id (int): bm25 score (float)} }
        inverted_index (dict): A dictionary mapping the token to a list that contains the number of documents with that token as the first element,
                                and a list of tuples as the second element, where each tuple contains the document ID and the corresponding count of 
                                the token in that document, for each document that contains that token.
                                { token (str): [number of documents with token (int), [( document ID (int), token count (int) )]] (List[int, List[tuple]]) }
        
    
    Returns:
        list: A list of tuples in the form (document_id, BM25_score), sorted by BM25_score in descending order.
    """
    # Initialize default BM25 score to 0 for all documents
    scores = {document_id: 0 for document_id, _ in documents.items()}

    # Compute aggregated BM25 scores by only considering the documents that contain a term
    for term in query:
        # Skip terms that are not in our vocabulary
        if term not in inverted_index:
            continue
        # Only process documents that contain the term
        containing_documents = inverted_index[term][1]
        for document_id, _ in containing_documents:
            scores[document_id] += bm25_matrix[term][document_id]

    # Compute final sorted ranking
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)

#################################
# Retrieval and Ranking Pipeline
#################################

def load_and_rank():
    """Loads and ranks documents using BM25, then reranks with the `OpenMatch/cocodr-large-msmarco` model, for each query."""

    # Read in the query corpus and preprocess

    # Dictionary representing the queries by query id
    # { query_id (int): list of tokens (list) }
    queries = dict()
    # { query_id (int): unprocessed content of query }
    queries_unprocessed = dict()

    # Read in queries
    with open(query_path, 'r', encoding="utf-8") as query_corpus:
        for query in query_corpus:
            # Load in the query in json format
            data = json.loads(query)
            # Preprocess query text before saving
            queries[data["_id"]] = preprocess_query(query)
            # Extract the document title and text before assigning to dictionary
            queries_unprocessed[data["_id"]] = extract_query_text(query)

    # Read in the document corpus and preprocess (step 1)
    start_time = time.time()

    # Dictionary representing the corpus by document id
    # { document_id (int): ( number of tokens (int), list of tokens (list) ) (tuple) }
    documents = dict()
    # { document_id (int): unprocessed content of document }
    documents_unprocessed = dict()

    # Read in corpus
    with open(corpus_path, 'r', encoding="utf-8") as corpus:
        for document in corpus:
            # Load in the document in json format
            data = json.loads(document)
            # Preprocess document title and text before assigning to dictionary
            documents[data["_id"]] = preprocess_document(document)
            # Extract the document title and text before assigning to dictionary
            documents_unprocessed[data["_id"]] = extract_document(document)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Preprocessing time: {elapsed_time: .4f} seconds")

    # Create inverted index (step 2)
    start_time = time.time()

    inverted_index = create_inverted_index(documents)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Inverted index creation time: {elapsed_time: .4f} seconds")

    # Retrieval and ranking (step 3)
    start_time = time.time()

    doc_length_sum = 0
    for document_id, _ in documents.items():
        doc_length_sum += documents[document_id][0]
    avdl = doc_length_sum/len(documents)

    # Perform retrieval and ranking
    # Compute the BM25 score between every document and every term in the vocabulary
    matrix = compute_bm25_matrix(documents, inverted_index, avdl)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"BM25 matrix creation time: {elapsed_time: .4f} seconds")

    # Create the transformer model and tokenizer
    model = AutoModel.from_pretrained("OpenMatch/cocodr-large-msmarco")
    tokenizer = AutoTokenizer.from_pretrained("OpenMatch/cocodr-large-msmarco")
    model.to(torch.device(device))
    model.eval()

    # Make results directory
    os.makedirs("results", exist_ok=True)

    # Dictionary to cache embeddings
    cached_embeddings = dict()

    # Write the top 100 ranked documents for every test query to an output file
    start_time = time.time()

    def embed(text):
        """Helper function that creates the embedding for some provided text using `OpenMatch/cocodr-large-msmarco`."""
        inputs = tokenizer(text, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            return model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, :1].squeeze() # the embedding of the [CLS] token after the final layer

    with open(f"results/Results_cocodr-large-msmarco.txt", "w") as file:
        for query_id, query_content in tqdm(queries.items(), desc=f"Query progress using cocodr-large-msmarco"):
            # Obtain the ranked documents for the current query
            ranked_documents = rank(query_content, documents, matrix, inverted_index)
            
            # { document_id (int): cosine similarity between document and query based on embedding (int) }
            similarities = dict()

            # Encode the query using the transformer model
            embedded_query = embed(queries_unprocessed[query_id])

            # Take the top 100 documents and rerank them using the transformer
            for i in range(100):
                document_id = ranked_documents[i][0]

                # Use the transformer to encode the documents into vectors incorporating semantic information
                # Note that we use the unprocessed version of the documents (to allow for semantic analysis)
                if document_id not in cached_embeddings:
                    embedding = embed(documents_unprocessed[document_id])
                    cached_embeddings[document_id] = embedding
                else:
                    embedding = cached_embeddings[document_id]

                # Compute cosine similarity between encoded document and query
                similarity = F.cosine_similarity(embedded_query, embedding, dim=0)
                similarities[document_id] = similarity.item()

            # Rerank based on cosine similarities by
            # sorting the dictionary in descending order by the items and storing it in a list
            sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)

            # Write results to the file
            for i in range(100):
                document_id = sorted_similarities[i][0]
                document_rank = i + 1
                document_score = sorted_similarities[i][1]
                tag = "cocodr-large-msmarco"
                file.write(f"{str(query_id)} Q0 {str(document_id)} {str(document_rank)} {str(document_score)} {tag}\n")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Query ranking time for cocodr-large-msmarco: {elapsed_time: .4f} seconds")

##############
# Entry Point
##############

if __name__ == "__main__":
    # Path Parameters
    base_dir = sys.argv[1] if len(sys.argv) >= 2 else "./scifact/"

    corpus_filename = sys.argv[2] if len(sys.argv) >= 3 else "corpus.jsonl"
    corpus_path = base_dir + corpus_filename

    query_filename = sys.argv[3] if len(sys.argv) >= 4 else "queries.jsonl"
    query_path = base_dir + query_filename

    # Load the documents, create inverted index, and run ranking algorithm
    load_and_rank()