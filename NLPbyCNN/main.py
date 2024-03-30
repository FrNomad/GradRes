import numpy as np
import torch
import nltk
from torch.utils.data import DataLoader, TensorDataset
from nltk.tokenize import word_tokenize
from network import *
from preprocess import *

X_data, y_data = read_file("data/imdb_ds.csv")

############################################################## //read file

X_train, X_valid, X_test, y_train, y_valid, y_test = split_dataset(X_data, y_data)

############################################################## //split dataset

tokenized_X_train = tokenize(X_train)
tokenized_X_test = tokenize(X_test)
tokenized_X_valid = tokenize(X_valid)

############################################################## //tokenize

vocab = generate_vocab(tokenized_X_train, threshold=3)

############################################################## //stop word removal

word_to_index, index_to_word, vocab_size = indexing_vocab(vocab)

max_len = 500

padded_X_train = encode_sentence(tokenized_X_train, word_to_index, max_len=max_len)
padded_X_valid = encode_sentence(tokenized_X_valid, word_to_index, max_len=max_len)
padded_X_test = encode_sentence(tokenized_X_test, word_to_index, max_len=max_len)

############################################################## //encoding

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

train_label_tensor = torch.tensor(np.array(y_train))
valid_label_tensor = torch.tensor(np.array(y_valid))
test_label_tensor = torch.tensor(np.array(y_test))

encoded_train = torch.tensor(padded_X_train).to(torch.int64)
train_dataset = TensorDataset(encoded_train, train_label_tensor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)

encoded_test = torch.tensor(padded_X_test).to(torch.int64)
test_dataset = TensorDataset(encoded_test, test_label_tensor)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=1)

encoded_valid = torch.tensor(padded_X_valid).to(torch.int64)
valid_dataset = TensorDataset(encoded_valid, valid_label_tensor)
valid_dataloader = DataLoader(valid_dataset, shuffle=True, batch_size=1)

total_batch = len(train_dataloader)
print("Total batch count: {}".format(total_batch))

############################################################## //set dataloader


def calculate_accuracy(logits, labels) :
    predicted = torch.argmax(logits, dim=1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct/total
    return accuracy

def evaluate(model, valid_dataloader, criterion, device) :
    val_loss, val_correct, val_total = 0, 0, 0

    model.eval()
    with torch.no_grad() :
        for batch_X, batch_y in valid_dataloader :
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            val_loss += loss.item()
            val_correct += calculate_accuracy(logits, batch_y) * batch_y.size(0)
            val_total += batch_y.size(0)

    val_accuracy = val_correct/val_total
    val_loss /= len(valid_dataloader)

    return val_loss, val_accuracy 


############################################################## //evaluate functions


learning_rate = 1e-3
num_epochs = 5
model_path = "models/best_model_checkpoint.pth"

model = CNN(vocab_size, num_labels=len(set(y_train)))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


best_val_loss = float('inf')

for epoch in range(num_epochs) :
    train_loss, train_correct, train_total = 0, 0, 0

    model.train()
    for batch_X, batch_y in train_dataloader :
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        logits = model(batch_X)
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += calculate_accuracy(logits, batch_y) * batch_y.size(0)
        train_total += batch_y.size(0)

    train_accuracy = train_correct/train_total
    train_loss /= len(train_dataloader)
    val_loss, val_accuracy = evaluate(model, valid_dataloader, criterion, device)

    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    if val_loss < best_val_loss :
        print(f'Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Checkpoint saved!')
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_path)


############################################################## //load, learn model


model.load_state_dict(torch.load(model_path))
model.to(device)

val_loss, val_accuracy = evaluate(model, valid_dataloader, criterion, device)

print(f'Best model validation loss: {val_loss:.4f}')
print(f'Best model validation accuracy: {val_accuracy:.4f}')

test_loss, test_accuracy = evaluate(model, test_dataloader, criterion, device)

print(f'Best model test loss: {test_loss:.4f}')
print(f'Best model test accuracy: {test_accuracy:.4f}')

############################################################## //evaluate model


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