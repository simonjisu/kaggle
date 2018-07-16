import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.data import Field, BucketIterator, TabularDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from bidirecGRU import bidirec_GRU
USE_CUDA = torch.cuda.is_available()
DEVICE = 0 if USE_CUDA else -1

PHRASE = Field(tokenize=str.split, use_vocab=True, lower=True, include_lengths=True,
               batch_first=True)
SENT = Field(sequential=False, use_vocab=False, preprocessing=lambda x: int(x))

train_data, valid_data = TabularDataset.splits(
       path='./', train='train_data.txt', validation="valid_data.txt", 
       format='tsv', fields=[('phrase', PHRASE), ('sent', SENT)])

PHRASE.build_vocab(train_data)

BATCH = 64

train_loader, valid_loader = BucketIterator.splits(
    (train_data, valid_data), batch_size=BATCH, device=DEVICE,
    sort_key=lambda x: len(x.phrase), sort_within_batch=True, repeat=False)

V = len(PHRASE.vocab)
D = 100
H = 300
H_f = 1000
O = 5
DA = 300
R = 10
N_LAYERS = 1
bidirec = True
weight_decay_rate = 0.0001
LR = 0.01
STEP = 30

model = bidirec_GRU(V, D, H, H_f, O, DA, R, 
                    num_layers=N_LAYERS, bidirec=bidirec, use_cuda=USE_CUDA)
if USE_CUDA:
    model = model.cuda()
    
loss_function = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.0001)
scheduler = optim.lr_scheduler.MultiStepLR(gamma=0.1, milestones=[10, 20, 30], optimizer=optimizer)

valid_losses = [10e5]
model.train()
for step in range(STEP):
    losses=[]
    scheduler.step()
    # train
    for batch in train_loader:
        inputs, lengths = batch.phrase
        targets = batch.sent
        if 0 in lengths:
            inputs = inputs[lengths.ne(0)]
            targets = targets[lengths.ne(0)]
            lengths = lengths[lengths.ne(0)]
        
        model.zero_grad()
        
        preds, loss_P = model(inputs, lengths)
        
        loss = loss_function(preds, targets) + loss_P
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
        optimizer.step()
    
    # valid
    model.eval()
    valid_loss = []
    for batch in valid_loader:
        inputs, lengths = batch.phrase
        targets = batch.sent
        if 0 in lengths:
            inputs = inputs[lengths.ne(0)]
            targets = targets[lengths.ne(0)]
            lengths = lengths[lengths.ne(0)]
        
        preds, _ = model(inputs, lengths)
        v_loss = loss_function(preds, targets) #+ loss_P
        valid_loss.append(v_loss.item())
        
    valid_losses.append(np.mean(valid_loss))
    
    if step >= 10 & (valid_losses[-2] - valid_losses[-1] < 0):
        torch.save(model.state_dict(), './model/model_2({}-{:.4f})'.format(step, valid_losses[-1]))
        
    string = '[{}/{}] train_loss: {:.4f}, valid_loss: {:.4f}, lr: {:.4f}'.format(
        step+1, STEP, np.mean(losses), np.mean(valid_loss), scheduler.get_lr()[0])
    print(string)
    losses = []
    valid_loss = []
    model.train()