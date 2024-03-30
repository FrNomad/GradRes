import numpy as np
import torch
import time
from torch.utils.data import DataLoader, TensorDataset
from nltk.tokenize import word_tokenize
from network import *
from preprocess import *


X_data, y_data = read_file("data/imdb_ds.csv")

############################################################## //read file

X_train, _, X_test, y_train, _, y_test = split_dataset(X_data, y_data)

############################################################## //split dataset

tokenized_X_train = tokenize(X_train)
tokenized_X_test = tokenize(X_test)

############################################################## //tokenize

vocab = generate_vocab(tokenized_X_train, threshold=3)

############################################################## //stop word removal

word_to_index, index_to_word, vocab_size = indexing_vocab(vocab)

############################################################## //encoding


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

model_path = "models/best_model_checkpoint.pth"
model = CNN(vocab_size, num_labels=len(set(y_train)))
model.load_state_dict(torch.load(model_path))
model.to(device)

index_to_tag = {0: "NEGATIVE", 1: "POSITIVE"}

def predict(text, model, word_to_index, index_to_tag) :

    model.eval()

    tokens = word_tokenize(text)
    token_indices = [word_to_index.get(token.lower(), 1) for token in tokens]

    input_tensor = torch.tensor([token_indices], dtype=torch.long).to(device)

    with torch.no_grad() :
        logits = model(input_tensor)
    
    _, predicted_index = torch.max(logits, dim=1)
    predicted_tag = index_to_tag[predicted_index.item()]

    return predicted_tag



while True :
    mode = int(input("Enter mode (0: evaluate randomly, 1: evaluate user) >> "))
    if mode in (0, 1) :
        break

if mode == 0 :
    while True :
        waiting = input()
        print("=> %s\n"%(predict(X_data.sample(n=1, random_state=int(time.time())).values.tolist()[0][0], model, word_to_index, index_to_tag)))
elif mode == 1 :
    while True :
        test_input = input(">> ")
        print("=> %s\n"%(predict(test_input, model, word_to_index, index_to_tag)))