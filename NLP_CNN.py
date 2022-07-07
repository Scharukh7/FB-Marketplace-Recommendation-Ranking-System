from turtle import forward
from unicodedata import category
import pandas as pd
import numpy as np
from sqlalchemy import desc
import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


products = '/Users/paddy/Desktop/AiCore/facebook_ml/nlp_fin.pkl'

df =  pd.read_pickle(products)
df.head

#%%
# %%
class productsPreProcessing(Dataset):
    def __init__(self):
        super().__init__()
        df =  pd.read_pickle(products)
        self.descriptions = df['product_description']
        self.categories = df['category']
        self.max_seq_len = 100
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = self.get_vocab()
        self.descriptions = self.tokenize_descriptions(self.descriptions)



    def get_vocab(self):
        def yield_tokens():
            for description in self.descriptions:
                tokens = self.tokenizer(description)
                yield tokens
        token_generator = yield_tokens()

        vocab = build_vocab_from_iterator(token_generator, specials=['<UNK>'])
        print('length of vocab:', len(vocab))
        return vocab


    def tokenize_descriptions(self, descriptions):
        def tokenize_description(description):
            words = self.tokenizer(description)
            words = words[:100]
            pad_length = self.max_seq_len - len(words)
            words.extend(['<UNK>']*pad_length)
            tokenized_desc = self.vocab(words)
            tokenized_desc = torch.tensor(tokenized_desc)
            return tokenized_desc

        descriptions = descriptions.apply(tokenize_description)
        return descriptions


    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        description = self.descriptions.iloc[idx]
        category = self.categories.iloc[idx]
        return (description, category)



dataset = productsPreProcessing()

print(dataset[2])
#%%



class CNN(torch.nn.Module):
    def __init__(self, pretrained_weights=None):
        super().__init__()
        no_words = 26888
        embedding_size = 16
        self.embedding = torch.nn.Embedding(no_words, embedding_size)
        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(embedding_size, 32, 2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(98, 128),
            torch.nn.Softmax()
        )


    def forward(self, X):
        # print(X.shape)
        X = self.embedding(X)
        X = X.transpose(2, 1)
        # print(X.shape)
        # print(X)
        return self.layers(X)

cnn = CNN()


# example = dataset[1]
# description, category = example
# prediction = cnn(description.unsqueeze(0))
# print(prediction)
# print(category)



train_split = 0.7
validation_split = 0.15
batch_size = 64

data_size = len(dataset)
print(f'dataset contains {data_size} Images')

train_size = int(train_split * data_size)
val_size = int(validation_split * data_size)
test_size = data_size - (val_size + train_size)
train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_samples = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_samples = DataLoader(val_data, batch_size=batch_size)
test_samples = DataLoader(test_data, batch_size=batch_size)




#%%
def train_model(model, epochs):
    writer = SummaryWriter()
    model.train()
    print('training model')
    optimiser = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        for phase in [train_samples, val_samples]:
            if phase == train_samples:
                print('training')
            else:
                print('val')
            for i, (features, labels) in enumerate(phase):
                num_correct = 0
                num_samples = 0
                print(features)
                print(labels)
                # features, labels = features, labels
                # features = features.to(device)  # move to device
                # labels = labels.to(device)
                predict = model(features.unsqueeze(0))
                # predict = predict.type(torch.LongTensor)
                print('this is label:', labels)
                print(labels.shape)
                print('this is predict:', predict)
                print(predict.shape)
                labels = labels
                loss = F.cross_entropy(predict, labels)
                _, preds = predict.max(1)
                num_correct += (preds == labels).sum()
                num_samples += preds.size(0)
                acc = float(num_correct) / num_samples
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()
                # writer.add_scalar('Loss', loss, epoch)
                # writer.add_scalar('Accuracy', acc, epoch)
                if i % 10 == 9:
                    if phase == train_samples:
                      writer.add_scalar('Training Loss', loss, epoch)
                      writer.add_scalar(' Training Accuracy', acc, epoch)
                      print('training_loss')
                    else:
                      writer.add_scalar('Validation Loss', loss, epoch)
                      writer.add_scalar('Validation Accuracy', acc, epoch)
                      print('val_loss') 
                    # print(batch) # print every 50 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss}')
                    print(f'Got {num_correct} / {num_samples} with accuracy: {acc * 100}%')
                    writer.flush()
            

train_model(cnn, 50)