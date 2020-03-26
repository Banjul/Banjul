import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


    
class bilstm(torch.nn.Module):
    def __init__(self, batch_size, output_size, hidden_size,vocab_size, 
                 embed_dim, bidirectional, dropout, sequence_length, 
                 pre_trained,pre_trained_path,freeze):
        super(bilstm, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.pre_trained = pre_trained
        self.pre_trained_path = pre_trained_path
        self.freeze = freeze
        #Using pre-trained embeddings
        if self.pre_trained:
            print('Using pre-trained word embeddings')
            gmat = []
            gdict2 = {}
            for h, line in enumerate(open(pre_trained_path, 'r').readlines()):
                line = line.strip()
                line = line.split()
                word = line[0]
                gdict2[word] = h
                vector = [float(item) for item in line[1:]]
                gmat.append(vector)
        
            weight = torch.FloatTensor(gmat)
            if not self.freeze:
                print('Freeze! Fine tuning')
            self.lookup_table = nn.Embedding.from_pretrained(weight,freeze = self.freeze, padding_idx=0)
        else:
            print('Using randomly initialize word embeddings')
            self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        
        self.lookup_table.weight.data.uniform_(-1., 1.)

        self.layer_size = 1
        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_size,
                            self.layer_size,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)

        if self.bidirectional:
            self.layer_size = self.layer_size * 2
        else:
            self.layer_size = self.layer_size


        self.label = nn.Linear(hidden_size * self.layer_size, output_size)
        


    def forward(self, input_sentences, batch_size=None):
        input = self.lookup_table(input_sentences)
        input = input.permute(1, 0, 2)


        h_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))
        c_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))

        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        final = lstm_output[-1]
        logits = self.label(final)
        return logits

class bow_nn(torch.nn.Module):
    def __init__(self, input_size, hidden_size,output_size,
                 emb_dim,pre_trained,pre_trained_path, freeze):
        super(bow_nn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.emb_dim = emb_dim
        self.pre_trained = pre_trained
        self.pre_trained_path = pre_trained_path
        self.freeze = freeze
        if self.pre_trained:
            print('Using pre-trained word embeddings')
            gmat = []
            gdict2 = {}
            for h, line in enumerate(open(self.pre_trained_path, 'r').readlines()):
                line = line.strip()
                line = line.split()
                word = line[0]
                gdict2[word] = h
                vector = [float(item) for item in line[1:]]
                gmat.append(vector)       
            weight = torch.FloatTensor(gmat)
            if not self.freeze:
                print('Freeze! Fine tuning')
            self.lookuptable = nn.Embedding.from_pretrained(weight,freeze = self.freeze, padding_idx=0)
        else:
            print('Using randomly initialize word embeddings')
            self.lookuptable = nn.Embedding(self.emb_dim, 300)
        
        self.first_linear= nn.Linear(self.input_size, self.hidden_size)
        self.second_linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self,  sentences, lookuptabel, word_to_ix):
        sents_vec = []
        for sentence in sentences:
            sent_vec = torch.zeros([1, 300], dtype=torch.float)
            for word in sentence:
                if word in word_to_ix.keys():
                    ix = word_to_ix[word]#word_to_ix是固定的，只用train数据生成一次，同理dictionary，label_to_id
                    ix = torch.tensor(ix)
                    word_vec = lookuptabel(ix)#lookuptable 每次训练都会更形
                else:
                    ix = word_to_ix['#UNK#']
                    ix = torch.tensor(ix)
                    word_vec = lookuptabel(ix)
                sent_vec = torch.add(sent_vec, word_vec)
            sent_vec = np.array(sent_vec.data)
            sents_vec.append(sent_vec[0])
        sents_vec = np.array(sents_vec)
        x= torch.from_numpy(sents_vec)
        x=self.first_linear(x)
        x= F.relu(x)#激活函数
        x= self.second_linear(x)
        return x