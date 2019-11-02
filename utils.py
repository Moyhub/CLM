import random, torch, copy, math
import numpy as np
import torch.nn.init as init
from torch.nn.utils import weight_norm

from collections import Counter

class Initializer(object):
    def __init__(self):
        pass

    def init_weight_(self, weight):
        init.kaiming_uniform_(weight)

    def init_embedding_(self, input_embedding):
        """
        Initialize embedding
        """
        # init_weight_(input_embedding.weight)
        # init.normal_(input_embedding.weight, 0, 0.1)
        bias = np.sqrt(3.0 / input_embedding.weight.size(1))
        init.uniform_(input_embedding.weight, -bias, bias)

    def init_linear_(self, input_linear, in_features, dropout):
        """
        Initialize linear transformation
        """
        init.normal_(input_linear.weight, mean=0, std=math.sqrt((1 - dropout) / in_features))
        # init_weight_(input_linear.weight)
        if input_linear.bias is not None:
            # input_linear.bias.data.zero_()
            init.constant_(input_linear.bias, 0)

        weight_norm(input_linear)

    def init_lstm_(self, input_lstm):
        """
        Initialize lstm
        """
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind))
            self.init_weight_(weight)

            weight = eval('input_lstm.weight_hh_l' + str(ind))
            self.init_weight_(weight)
        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
                self.init_weight_(weight)

                weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
                self.init_weight_(weight)

        if input_lstm.bias:
            for ind in range(0, input_lstm.num_layers):
                weight = eval('input_lstm.bias_ih_l' + str(ind))
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                weight = eval('input_lstm.bias_hh_l' + str(ind))
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            if input_lstm.bidirectional:
                for ind in range(0, input_lstm.num_layers):
                    weight = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                    weight.data.zero_()
                    weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                    weight = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                    weight.data.zero_()
                    weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

class Corpus(object):
    def __init__(self, path):
        self.word2idx, self.idx2word, self.vocab_size = self._build_vocba(path)

        self.train = self._tokenize('{}/train.txt'.format(path))
        self.valid = self._tokenize('{}/testa.txt'.format(path))
        self.test = self._tokenize('{}/testb.txt'.format(path))

    def _build_vocba(self, path, min_count=5):
        count_chars = Counter()
        fns = ['{}/train.txt'.format(path), '{}/testa.txt'.format(path), '{}/testb.txt'.format(path)]
        for fn in fns:
            with open(fn, 'r', encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip().split()
                    count_chars.update(line)

        idx2word = ['<PAD>', '<UNK>'] + sorted([w for w, c in count_chars.items() if c >= min_count])
        word2idx = {word: idx for idx, word in enumerate(idx2word)}

        # save to file
        with open('{}/vocab.txt'.format(path), 'w', encoding='utf-8') as fout:
            for word in idx2word:
                fout.write(word + '\n')

        return word2idx, idx2word, len(idx2word)

    def _tokenize(self, fn_read):
        res = []
        with open(fn_read, 'r', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip().split()
                if len(line) <= 1:
                    continue
                words = []
                for word in line:
                    if word in self.word2idx:
                        words.append(self.word2idx[word])
                    else:
                        words.append(self.word2idx['<UNK>'])
                res.append(words)

        return res


# class Dictionary(object):
#     def __init__(self):
#         self.word2idx = {'<pad>': 0}
#         self.idx2word = ['<pad>']
#
#     def add_word(self, word):
#         if word not in self.word2idx:
#             self.word2idx[word] = len(self.idx2word)
#             self.idx2word.append(word)
#         return self.word2idx[word]
#
#     def get_length(self):
#         return len(self.idx2word)
#
#     def save(self, fn_save):
#         with open(fn_save, 'w', encoding='utf-8') as fout:
#             for word in self.idx2word:
#                 fout.write(word + '\n')
#
# class Corpus_(object):
#     def __init__(self, path):
#         self.dictionary = Dictionary()
#
#         self.train = self.tokenize('{}/train.txt'.format(path))
#         self.valid = self.tokenize('{}/testa.txt'.format(path))
#         self.test = self.tokenize('{}/testb.txt'.format(path))
#
#         self.dictionary.save('{}/vocab.txt'.format(path))
#
#     def tokenize(self, fn_read):
#         res = []
#         with open(fn_read, 'r', encoding='utf-8') as fin:
#             for line in fin:
#                 line = line.strip().split()
#                 if len(line) <= 1:
#                     continue
#                 words = []
#                 for word in line:
#                     word_id = self.dictionary.add_word(word)
#                     words.append(word_id)
#                 res.append(words)
#
#         return res

def create_batches(data, batch_size, max_len=20, order='random', device='cuda'):

    def pad_seq(seq, max_len, padding=0):
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq += [padding] * (max_len - len(seq))
        return seq

    newdata = copy.deepcopy(data) #是否打乱数据
    if order == 'sort':
        newdata.sort(key=lambda x: len(x))
    elif order == 'random':
        random.shuffle(newdata)

    batches = []
    num_batches = np.ceil(len(data) / float(batch_size)).astype('int') #取块的上限

    for i in range(num_batches):
        batch_data = newdata[(i*batch_size) : min(len(data), (i+1)*batch_size)] #分块
        batch_data.sort(key=lambda x: -1.*len(x))

        words_lengths = np.array([np.min([len(s), max_len]) for s in batch_data]) #单词长度限定
        batch_data_padded = np.array([pad_seq(s, np.min([np.max(words_lengths), max_len])) for s in batch_data])

        batch_data_tensor = torch.tensor(batch_data_padded[:, :-1], requires_grad=False, dtype=torch.long).to(device)# batch_size * seq_len
        batch_target_tensor = torch.tensor(batch_data_padded[:, 1:], requires_grad=False, dtype=torch.long).to(device)
        words_lengths_tensor = torch.tensor(words_lengths-1, requires_grad=False, dtype=torch.long).to(device)

        out_dict = {'data': batch_data_tensor,
                    'target': batch_target_tensor,
                    'words_lengths': words_lengths_tensor}

        batches.append(out_dict)

    return batches


def extract_words_list(fn_read, fn_save, type='entity'):
    """ Extract entities/nonentities from `fn_read`, then save to `fn_save` """
    print('Extract {} from {}...'.format(type, fn_read))

    res = []
    with open(fn_read, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip().split()
            if len(line) == 0:
                continue
            word = line[0]
            label = line[-1]
            if type == 'entity':
                if label != 'O' and len(word) > 1: # for entity
                    res.append(word)
                    res.append(word.capitalize())
            else:
                if label == 'O': # for non-entity
                    res.append(word)
                    res.append(word.capitalize())

    res = set(res)
    with open(fn_save, 'w', encoding='utf-8') as fout:
        for item in sorted(res):
            item = list(item)
            fout.write(' '.join(item) + '\n')


def merge_and_clean_dataset(fn_read_list, fn_save):
    res = []
    for fn_read in fn_read_list:
        with open(fn_read, 'r', encoding='utf-8') as fin:
            for line in fin:
                word = ''.join(line.strip().split())
                res.append(word)
    num_init = len(res)
    res = sorted(set(res))
    num_clean = len(res)
    with open(fn_save, 'w', encoding='utf-8') as fout:
        for word in res:
            fout.write(' '.join(list(word)) + '\n')
    print('{}/{} saved.'.format(num_clean, num_init))


if __name__ == '__main__':
    flag = False
    if flag:
        types = ['train', 'testa', 'testb']
        for type in types:
            extract_words_list('../data/CoNLL2003/eng.{}.bio'.format(type), 'eng_nonentity/{}.txt'.format(type), type='nonentity')
            # extract_words_list('../data/CoNLL2003/deu.{}.bio'.format(type), 'deu_nonentity/{}.txt'.format(type), type='nonentity')
            # extract_words_list('../data/CoNLL2002/esp.{}.bio'.format(type), 'esp_nonentity/{}.txt'.format(type), type='nonentity')
            # extract_words_list('../data/CoNLL2002/ned.{}.bio'.format(type), 'ned_nonentity/{}.txt'.format(type), type='nonentity')
            #
            extract_words_list('../data/CoNLL2003/eng.{}.bio'.format(type), 'eng_entity/{}.txt'.format(type), type='entity')
            # extract_words_list('../data/CoNLL2003/deu.{}.bio'.format(type), 'deu_entity/{}.txt'.format(type), type='entity')
            # extract_words_list('../data/CoNLL2002/esp.{}.bio'.format(type), 'esp_entity/{}.txt'.format(type), type='entity')
            # extract_words_list('../data/CoNLL2002/ned.{}.bio'.format(type), 'ned_entity/{}.txt'.format(type), type='entity')

    flag = False
    if flag:
        # clean English dataset.
        merge_and_clean_dataset('eng_entity/train.txt', 'eng_entity/train_cleaned.txt')
        merge_and_clean_dataset('eng_entity/testa.txt', 'eng_entity/testa_cleaned.txt')
        merge_and_clean_dataset('eng_entity/testb.txt', 'eng_entity/testb_cleaned.txt')
        merge_and_clean_dataset('eng_nonentity/train.txt', 'eng_nonentity/train_cleaned.txt')
        merge_and_clean_dataset('eng_nonentity/testa.txt', 'eng_nonentity/testa_cleaned.txt')
        merge_and_clean_dataset('eng_nonentity/testb.txt', 'eng_nonentity/testb_cleaned.txt')

    flag = True
    if flag:
        merge_and_clean_dataset(['eng_entity/train.txt', 'esp_entity/train.txt'], 'eng_esp_entity/train.txt')
        merge_and_clean_dataset(['eng_entity/train.txt', 'ned_entity/train.txt'], 'eng_ned_entity/train.txt')
        merge_and_clean_dataset(['eng_entity/train.txt', 'deu_entity/train.txt'], 'eng_deu_entity/train.txt')
        merge_and_clean_dataset(['eng_nonentity/train.txt', 'esp_nonentity/train.txt'], 'eng_esp_nonentity/train.txt')
        merge_and_clean_dataset(['eng_nonentity/train.txt', 'ned_nonentity/train.txt'], 'eng_ned_nonentity/train.txt')
        merge_and_clean_dataset(['eng_nonentity/train.txt', 'deu_nonentity/train.txt'], 'eng_deu_nonentity/train.txt')

        merge_and_clean_dataset(['eng_entity/testa.txt', 'esp_entity/testa.txt'], 'eng_esp_entity/testa.txt')
        merge_and_clean_dataset(['eng_entity/testa.txt', 'ned_entity/testa.txt'], 'eng_ned_entity/testa.txt')
        merge_and_clean_dataset(['eng_entity/testa.txt', 'deu_entity/testa.txt'], 'eng_deu_entity/testa.txt')
        merge_and_clean_dataset(['eng_nonentity/testa.txt', 'esp_nonentity/testa.txt'], 'eng_esp_nonentity/testa.txt')
        merge_and_clean_dataset(['eng_nonentity/testa.txt', 'ned_nonentity/testa.txt'], 'eng_ned_nonentity/testa.txt')
        merge_and_clean_dataset(['eng_nonentity/testa.txt', 'deu_nonentity/testa.txt'], 'eng_deu_nonentity/testa.txt')

        merge_and_clean_dataset(['eng_entity/testb.txt', 'esp_entity/testb.txt'], 'eng_esp_entity/testb.txt')
        merge_and_clean_dataset(['eng_entity/testb.txt', 'ned_entity/testb.txt'], 'eng_ned_entity/testb.txt')
        merge_and_clean_dataset(['eng_entity/testb.txt', 'deu_entity/testb.txt'], 'eng_deu_entity/testb.txt')
        merge_and_clean_dataset(['eng_nonentity/testb.txt', 'esp_nonentity/testb.txt'], 'eng_esp_nonentity/testb.txt')
        merge_and_clean_dataset(['eng_nonentity/testb.txt', 'ned_nonentity/testb.txt'], 'eng_ned_nonentity/testb.txt')
        merge_and_clean_dataset(['eng_nonentity/testb.txt', 'deu_nonentity/testb.txt'], 'eng_deu_nonentity/testb.txt')
