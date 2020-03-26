import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import Counter

def word2idx(sents, word2idx):
    return [[word2idx[w] if w in word2idx else UNK for w in s] for s in sents]


class Dictionary(object):
    def __init__(self, word2idx={}, idx_num=0):
        self.word2idx = word2idx
        self.idx = idx_num

    def _add(self, word):
        if self.word2idx.get(word) is None:
            self.word2idx[word] = self.idx
            self.idx += 1

    def _convert(self):
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return self.idx

    def __str__(self):
        return "%s(size = %d)".format(self.__class__.__name__, len(self.idx))

PAD = 0
UNK = 1

WORD = {
    UNK: '<unk>',
    PAD: '<pad>'
}
stop_and_punctuations = ["a", "an", "the","in", "out", "on", "off", "over", "under","but",
                         "can","?",",","\'","`"]
class Words(Dictionary):
    def __init__(self):
        word2idx = {
            WORD[PAD]: PAD,
            WORD[UNK]: UNK
        }
        super().__init__(word2idx=word2idx, idx_num=len(word2idx))

    def __call__(self, sents):
        words = [word for sent in sents for word in sent]
        for word in words:
            if word not in stop_and_punctuations:
                self._add(word)

class Labels(Dictionary):
    def __init__(self):
        super().__init__()

    def __call__(self, labels):
        _labels = labels
        for label in _labels:
            self._add(label)

class Corpus(object):
    def __init__(self, path, save_data, max_len=16):
        self.train = os.path.join(path, "train.txt")
        self.valid = os.path.join(path, "dev.txt")
        self.test = os.path.join(path, "test.txt")
        self._save_data = save_data

        self.w = Words()
        self.l = Labels()
        self.max_len = max_len

    def parse_data(self, _file, is_train=True, is_valid = True,fine_grained=True):
        """
        fine_grained: Whether to use the fine-grained (50-class) version of TREC
                or the coarse grained (6-class) version.
        """
        _sents, _labels = [], []
        for sentence in open(_file):
            label, _, _words = sentence.replace('\xf0', ' ').partition(' ')
            label = label.split(":")[0] if not fine_grained else label

            word = _words.lower()  # [preprocessing] lowercase words
            
            words = word.strip().split()

            if len(words) > self.max_len:
                words = words[:self.max_len]

            _sents += [words]
            _labels += [label]
        if is_train:
            self.w(_sents)
            self.l(_labels)
            self.train_sents = _sents
            self.train_labels = _labels
        elif is_valid:
            self.valid_sents = _sents
            self.valid_labels = _labels
        else:
            self.test_sents = _sents
            self.test_labels = _labels

    def save(self):
        self.parse_data(self.train)
        self.parse_data(self.valid, False)
        self.parse_data(self.test,False,False)

        data = {
            'max_len': self.max_len,
            'dict': {
                'train': self.w.word2idx,
                'vocab_size': len(self.w),
                'label': self.l.word2idx,
                'label_size': len(self.l),
            },
            'train': {
                'src': word2idx(self.train_sents, self.w.word2idx),
                'label': [self.l.word2idx[l] for l in self.train_labels]
            },
            'dev': {
                'src': word2idx(self.valid_sents, self.w.word2idx),
                'label': [self.l.word2idx[l] for l in self.valid_labels]
            },
            'test': {
                'src': word2idx(self.test_sents, self.w.word2idx),
                'label': [self.l.word2idx[l] for l in self.test_labels],
                'label_words':self.test_labels
            }
        }

        torch.save(data, self._save_data)
        
        
class DataLoader(object):
    def __init__(self, src_sents, label, max_len,
                 batch_size=64, shuffle=False, evaluation=False):

        self.sents_size = len(src_sents)
        self._step = 0
        self._stop_step = self.sents_size // batch_size
        self.evaluation = evaluation

        self._batch_size = batch_size
        self._max_len = max_len
        self._src_sents = np.asarray(src_sents)
        self._label = np.asarray(label)
        if shuffle:
            self._shuffle()

    def _shuffle(self):
        indices = np.arange(self._src_sents.shape[0])
        np.random.shuffle(indices)
        self._src_sents = self._src_sents[indices]
        self._label = self._label[indices]

    def __iter__(self):
        return self

    def __next__(self):
        def pad_to_longest(insts, max_len):
            inst_data = np.array([inst + [0] * (max_len - len(inst)) for inst in insts])

            inst_data_tensor = Variable(torch.from_numpy(inst_data), volatile=self.evaluation)

            return inst_data_tensor

        if self._step == self._stop_step:
            self._step = 0
            raise StopIteration()

        _start = self._step*self._batch_size
        _bsz = self._batch_size
        self._step += 1
        data = pad_to_longest(self._src_sents[_start:_start+_bsz], self._max_len)
        label = Variable(torch.from_numpy(self._label[_start:_start+_bsz]),
                    volatile=self.evaluation)


        return data, label

class DataSplit(object):
    def load_split(self, filename):
        data = []
        sentences = []
        all_words = []
        labels = []
        with open(filename, encoding='ISO-8859-1') as file:
            for index, line in enumerate(file):
                words = line.split()
                t = (words[1:], words[0])
                data.append(t)
                all_words += words[1:]
                labels.append(words[0])
                sentences.append(words[1:])
        return data, sentences, all_words, labels

class dataProcessor(object):
    def word2index(self, train_data):
        word_to_ix = {}
        word_to_ix['#UNK#'] = 0
        for sent, _ in train_data:
            for word in sent:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)

        return word_to_ix
    '''
    获取label
    '''

    def count_labels(self, labels):
        unique_label=[]
        for letter in labels:
            if letter not in unique_label:
                unique_label.append(letter)

        return unique_label
    '''
    label索引
    '''

    def label2index(self,unique_label):
        label_to_ix={}
        index = 0
        for label in unique_label:
            label_to_ix[label] = index
            index += 1

        return label_to_ix

    '''
    统计出现的单词及其出现次数
    '''

    def freq_dic(self, all_words):
        diction={}
        cnt = Counter(all_words)
        for word, freq in cnt.items():
            diction[word] = [len(diction), freq]

        return diction
