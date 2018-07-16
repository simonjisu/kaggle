import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class bidirec_GRU(nn.Module):
    def __init__(self, V, D, H, H_f, O, da, r, num_layers=3, bidirec=False, use_cuda=False):
        """
        V: input_size = vocab_size
        D: embedding_size
        H: hidden_size
        H_f: hidden_size (fully-connected)
        O: output_size (fully-connected)
        da: attenion_dimension (hyperparameter)
        r: keywords (different parts to be extracted from the sentence)
        """
        super(bidirec_GRU, self).__init__()
        self.r = r
        self.da = da
        self.hidden_size = H
        self.num_layers = num_layers
        self.USE_CUDA = use_cuda
        self.num_directions = 2 if bidirec else 1
        
        self.embed = nn.Embedding(V, D)
        self.gru = nn.GRU(D, H, num_layers, batch_first=True, bidirectional=bidirec)
        self.attn = nn.Linear(self.num_directions*H, self.da, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.attn2 = nn.Linear(self.da, self.r, bias=False)
        self.attn_dist = nn.Softmax(dim=2)
        
        self.fc = nn.Sequential(
            nn.Linear(r*H*self.num_directions, H_f),
            nn.ReLU(),
            nn.Linear(H_f, O)
        )
            
    def init_GRU(self, batch_size):
        # (num_layers * num_directions, batch_size, hidden_size)
        hidden = torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size)
        if self.USE_CUDA:
            hidden = hidden.cuda()
        return hidden
    
    def penalization_term(self, A):
        """
        A : B, r, T
        Frobenius Norm 
        """
        eye = torch.eye(A.size(1)).expand(A.size(0), self.r, self.r) # B, r, r
        if self.USE_CUDA:
            eye = eye.cuda()
        P = torch.bmm(A, A.transpose(1, 2)) - eye # B, r, r
        loss_P = ((P**2).sum(1).sum(1) + 1e-10) ** 0.5
        loss_P = torch.sum(loss_P) / A.size(0)
        return loss_P
        
    def forward(self, inputs, inputs_lengths):
        """
        inputs: B, T, V
         - B: batch_size
         - T: max_len = seq_len
         - V: vocab_size
        inputs_lengths: length of each sentences
        """
        embed = self.embed(inputs)  # B, T, V  --> B, T, D
        hidden = self.init_GRU(inputs.size(0))  # num_layers * num_directions, B, H
        # pack sentences
        packed = pack_padded_sequence(embed, inputs_lengths.tolist(), batch_first=True)
        # packed: B * real_length, D
        output, hidden = self.gru(packed, hidden)
        # output: B * T, 2H
        # hidden: num_layers * num_directions, B, H
        
        # unpack sentences
        output, output_lengths = pad_packed_sequence(output, batch_first=True) 
        # output: B, T, 2H

        # Self Attention
        a1 = self.attn(output)  # Ws1(B, da, 2H) * output(B, T, 2H) -> B, T, da
        tanh_a1 = self.tanh(a1)  # B, T, da
        score = self.attn2(tanh_a1)  # Ws2(B, r, da) * tanh_a1(B, T, da) -> B, T, r
        self.A = self.attn_dist(score.transpose(1, 2))  # B, r, T
        self.M = self.A.bmm(output)  # B, r, T * B, T, 2H -> B, r, 2H 
        
        # Penalization Term
        loss_P = self.penalization_term(self.A)
        
        output = self.fc(self.M.view(self.M.size(0), -1)) # B, r, 2H -> resize to B, r*2H -> B, H_f -> Relu -> B, 1
        
        return output, loss_P
    
    def predict(self, inputs, inputs_lengths):
        preds, _ = self.forward(inputs, inputs_lengths)
        _, idx = F.softmax(preds, dim=1).max(1)
        return idx