import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split


def tokenize(sentences) :
    tokenized_sentences = []
    for sent in tqdm(sentences) :
        tokenized_sent = word_tokenize(sent)
        tokenized_sent = [word.lower() for word in tokenized_sent]
        tokenized_sentences.append(tokenized_sent)
    return tokenized_sentences

def texts_to_sequences(tokenized_X_data, word_to_index) :
    encoded_X_data = []
    for sent in tokenized_X_data :
        index_sequences = []
        for word in sent :
            try :
                index_sequences.append(word_to_index[word])
            except KeyError :
                index_sequences.append(word_to_index['<UNK>'])
        encoded_X_data.append(index_sequences)
    return encoded_X_data

def pad_sequences(sentences, max_len) :
    features = np.zeros((len(sentences), max_len), dtype=int)
    for index, sentence in enumerate(sentences) :
        if len(sentence) != 0 :
            features[index, :len(sentence)] = np.array(sentence)[:max_len]
    return features

def filter_stopword(word_counts, vocab, threshold) :
    total_cnt = len(word_counts)
    rare_cnt = 0

    for key, value in word_counts.items() :
        if value < threshold :
            rare_cnt += 1

    vocab_size = total_cnt - rare_cnt
    vocab = vocab[:vocab_size]
    print("Total number of filtered words: ", len(vocab))
    
    return vocab

############################################################## //internal

def read_file(path) :
    df = pd.read_csv(path)

    df['sentiment'] = df['sentiment'].replace(['positive','negative'],[1, 0])

    X_data = df['review']
    y_data = df['sentiment']
    print("Number of movie reviews: {}".format(len(X_data)))
    print("Number of labels: {}".format(len(y_data)))

    return X_data, y_data


def split_dataset(X_data, y_data) :
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
                                                    test_size=0.5,
                                                    random_state=0,
                                                    stratify=y_data)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                        test_size=0.2,
                                                        random_state=0,
                                                        stratify=y_train)
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def generate_vocab(X, threshold) :
    word_list = []
    for sent in X :
        for word in sent :
            word_list.append(word)

    word_counts = Counter(word_list)
    print("Total number of words: ", len(word_counts))

    vocab = sorted(word_counts, key=word_counts.get, reverse=True)

    return filter_stopword(word_counts, vocab, threshold)


def indexing_vocab(vocab) :
    word_to_index, index_to_word = {}, {}
    word_to_index['<PAD>'] = 0
    word_to_index['<UNK>'] = 1

    for index, word in enumerate(vocab) :
        word_to_index[word] = index + 2

    vocab_size = len(word_to_index)

    for key, value in word_to_index.items() :
        index_to_word[value] = key

    return word_to_index, index_to_word, vocab_size

def encode_sentence(tokenized_X_data, word_to_index, max_len) :
    return pad_sequences(texts_to_sequences(tokenized_X_data, word_to_index), max_len)