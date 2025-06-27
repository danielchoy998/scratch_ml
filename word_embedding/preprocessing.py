import requests
from bs4 import BeautifulSoup
import re
import numpy as np

"""
Preprocessing the text data
flow :
1. scrape the text from the url
2. text cleaning by removing references/citations and non-alphabetic characters
3. tokenize the text into words ie. [list of words] -> ['word1', 'word2', 'word3', ...]
4. build vocabulary from tokens ie. unique word dictionary
5. one-hot encode the words
"""

def scrape_text_from_url(url):
    """
    Fetches and cleans main text content from a web page.
    Args:
        url (str): The URL to scrape.
    Returns:
        str: Cleaned text content.
    """
    response = requests.get(url)
    # print(response) # A response object - 200 means success
    # print(response.text) # The HTML content of the page ie. HTML file

    soup = BeautifulSoup(response.text, 'html.parser') # initialise the BeautifulSoup object
    # Extract text from all paragraph tags
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    # Remove references/citations and non-alphabetic characters
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    print(type(text))
    return text.strip()

def tokenize_text(text):
    """
    Tokenize text into individual words (lowercase).
    Args:
        text (str): Raw text string.
    Returns:
        list: List of individual words.
    """
    tokens = text.lower().split()
    return tokens

def build_vocabulary(tokens):
    """
    Build vocabulary from tokens ie. unique word list and create word-to-index mappings for mapping the word.
    Args:
        tokens (list): List of words.
    Returns:
        tuple: (word2idx dict, idx2word dict, vocab_size int)
    """
    vocab = list(set(tokens))  # Get unique words
    vocab.sort()  # Sort for consistency
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word, len(vocab)

def one_hot_encode(word, word2idx, vocab_size):
    """
    Create one-hot vector for a given word.
    Args:
        word (str): The word to encode.
        word2idx (dict): Word to index mapping.
        vocab_size (int): Size of vocabulary.
    Returns:
        numpy.ndarray: One-hot vector of size vocab_size.
    """
    vector = np.zeros(vocab_size)
    if word in word2idx:
        # vector[word_id] = 1
        vector[word2idx[word]] = 1
    return vector


if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Natural_language_processing"
    
    # Step 1: Scrape raw text
    text = scrape_text_from_url(url)
    print("Raw text (first 200 chars):")
    print(f"'{text[:200]}...'\n")
    print(f"Type: {type(text)}")
    print(f"Length: {len(text)} characters\n")
    
    # Step 2: Tokenize into words
    tokens = tokenize_text(text)
    print("Tokenized words (first 20):")
    print(tokens[:20])
    print(f"Type: {type(tokens)}")
    print(f"Number of tokens: {len(tokens)}")
    print(f"Unique words: {len(set(tokens))}\n")
    
    # Step 3: Build vocabulary 
    word2idx, idx2word, vocab_size = build_vocabulary(tokens)
    print(f"Vocabulary size: {vocab_size}")
    print("First 10 words in vocabulary:", list(word2idx.keys())[:10])
    
    # Example: One-hot encode a few words
    sample_words = ["natural", "language", "processing"]
    print("\nOne-hot encoding examples:")
    for word in sample_words:
        if word in word2idx:
            vector = one_hot_encode(word, word2idx, vocab_size)
            print(f"'{word}' -> Index: {word2idx[word]}, Vector shape: {vector.shape}")
            print(f"First 10 elements: {vector[:10]}")
        else:
            print(f"'{word}' not found in vocabulary")
