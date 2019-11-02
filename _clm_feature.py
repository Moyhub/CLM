import argparse, utils, random, numpy
from collections import Counter
import torch
import matplotlib.pyplot as plt

class DataLoader(object):
    def __init__(self, fn_vocab):
        self.word2idx = {}
        self.idx2word = []
        self.load_vocab_(fn_vocab)
        self.vocab_size = len(self.word2idx)

    def load_vocab_(self, fn_vocab):
        with open(fn_vocab, 'r', encoding='utf-8') as fin:
            for word in fin:
                word = word.strip()
                assert len(word) > 0
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)

    def tokenize(self, word_list):
        """
        word_list: a list of word, each word is a list of chars.
        """
        res = []
        for word in word_list:
            word_id = []
            for s in word:
                if s in self.word2idx:
                    word_id.append(self.word2idx[s])
                else:
                    word_id.append(self.word2idx['<UNK>'])
            # word_id = [self.word2idx[s] for s in word if s in self.word2idx]
            if len(word_id) < 2:
                continue
            # assert len(word_id) >= 2

            res.append(word_id)

        return res


def load_conll_words(fn_conll):
    with open(fn_conll, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if line == '':
                continue

    # not finished yet
    pass

def load_clm_words(fn_clm_data):
    res = []
    with open(fn_clm_data, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip().split()
            res.append(line)

    return res

def evaluate(data_batches):
    model.eval()
    losses = []
    ntokens = loader.vocab_size
    with torch.no_grad():
        for i, batch_data in enumerate(data_batches):
            data = batch_data['data']
            targets = batch_data['target']
            words_lenghths = batch_data['words_lengths']

            output = model(data, words_lenghths)
            loss = criterion(output.view(-1, ntokens), targets.contiguous().view(targets.size(0)*targets.size(1))).item()

            losses.append(loss)

    return losses

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn_model', type=str, default='eng_esp_entity/model1.pt', help='path of the saved CLM')
    parser.add_argument('--fn_vocab', type=str, default='eng_esp_entity/vocab.txt', help='path of the vocabulary')

    parser.add_argument('--fn_data', type=str, default='eng_entity/train.txt+eng_entity/testb.txt+eng_nonentity/train.txt+eng_nonentity/testb.txt+esp_nonentity/train.txt+esp_entity/train.txt+esp_entity/testb.txt+esp_nonentity/train.txt+esp_nonentity/testb.txt', help='path of the dataset ot evaluate')

    parser.add_argument('--gpu_device', type=int, default=3, help='gpu device ID.')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()

    fn_model = args.fn_model
    fn_vocab = args.fn_vocab
    fn_data = args.fn_data.split('+')
    seed = args.seed

    gpu_device = args.gpu_device
    if gpu_device > 0:
        torch.cuda.set_device(gpu_device)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)

    # load model
    model = torch.load(fn_model, map_location={'cuda:1':'cuda:{}'.format(gpu_device)})
    # make the rnn parameters a continuous chunk, which will speed up forward pass
    model.rnn.flatten_parameters()
    criterion = torch.nn.CrossEntropyLoss()

    loader = DataLoader(fn_vocab)

    plt.figure()
    for fn in fn_data:
        # prepare dataset
        print('Processing %s...' % fn)
        word_list = load_clm_words(fn)
        test_data = loader.tokenize(word_list)
        test_data_batches = utils.create_batches(test_data, batch_size=1, device='cuda')

        losses = evaluate(test_data_batches)

        ppl_counter = Counter()

        x_interval = numpy.array([i*0.2 for i in range(100)])
        for loss in losses:
            idx = numpy.argmin(abs(x_interval-loss))
            ppl_counter.update([x_interval[idx]])

        keys = []
        vals = []
        for key, value in sorted(ppl_counter.items()):
            keys.append(key)
            vals.append(value)

        # keys = numpy.array(keys)
        vals = numpy.array(vals)
        vals = vals * 1.0 / numpy.sum(vals)

        plt.plot(keys, vals, label='data: {}'.format(fn))

    plt.legend()
    plt.xlabel('model: {}'.format(fn_model))
    plt.savefig('figs/{}_{}.png'.format(fn_model.split('/')[0], fn_model.split('/')[1]))