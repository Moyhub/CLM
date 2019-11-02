# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
import utils
import model
import logging, json, random

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')  #这句话是使用 main.py -h 所示的描述语句，同理下面的add_argument的help也会被显示出来
parser.add_argument('--data', type=str, default='./eng_esp_entity/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM', #help用来描述这个选项的作用 #type为参数类型
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=50,
                    help='size of word embeddings')  #词嵌入向量长度
parser.add_argument('--nhid', type=int, default=100,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate') #初始学习率
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',    #这里store_true十分有意思，就是python main.py时默认为False，python main.py --tied 注意没有赋值，这里认为是True
                    help='tie the word embedding and softmax weights')  #The word embedding and the softmax weights can be tied
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--gpu_device', type=int, default=6,
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--expt_id', type=str, default='0',
                    help='experiment id')
args = parser.parse_args()
###########################使用参数的方式#######################
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument()
# parser.parse_args()
# 对于参数dest='a'，则可以通过args.a访问该参数，相当于一个别名。
##############################################################

# save parameters
args.save = '{}model{}.pt'.format(args.data, args.expt_id)
params = vars(args)  #var是返回一个字典，params相当于一个dict,将args的Namespace转化成dict
with open('{}params{}.json'.format(args.data, args.expt_id), 'w', encoding='utf-8') as fw: #若想执行“w"操作，则params是dict形式则会出错，所以用json.dump转换为str并保存到json.
    json.dump(params, fw, indent=4, sort_keys=False)                #sort_keys是告诉编码器按照字典排序(a到z)输出。indent参数根据数据格式缩进显示，读起来更加清晰，4是缩进的位数
#这里我自己新建了文件，结果是params保存到文件中

# config logging
logging.basicConfig(filename='{}logging{}.log'.format(args.data, args.expt_id), level=logging.INFO)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)   #为CPU设置种子用于生成随机数，以使得结果是确定的
if args.gpu_device > 0:
    device = torch.device("cuda")
    torch.cuda.set_device(args.gpu_device) #id，默认从0开始使用GPU，这里是指定使用的GPU编号
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
else:
    device = torch.device("cpu")


# load data
corpus = utils.Corpus(args.data)
eval_batch_size = 10

train_data = utils.create_batches(corpus.train, args.batch_size, order='random', device=device) #为什么数据batch_size不同
val_data = utils.create_batches(corpus.valid, eval_batch_size, order='random', device=device)
test_data = utils.create_batches(corpus.test, eval_batch_size, order='random', device=device)


# build model
vocab_size = corpus.vocab_size  #idx2word的长度
model = model.RNNModel(args.model, vocab_size, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)
criterion = nn.CrossEntropyLoss()


# training code

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval() #训练模式model.eval() ：不启用 BatchNormalization 和 Dropout
    total_loss = 0.
    ntokens = corpus.vocab_size
    with torch.no_grad():
        for i, batch_data in enumerate(data_source):
            data = batch_data['data']  # batch_size * seq_len
            targets = batch_data['target']  # batch_size * seq_len
            words_lengths = batch_data['words_lengths']
            output = model(data, words_lengths)

            total_loss += criterion(output.view(-1, ntokens),
                                                targets.contiguous().view(targets.size(0) * targets.size(1))).item()
    return total_loss / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    #model.train() ：启用 BatchNormalization 和 Dropout
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = corpus.vocab_size
    for i, batch_data in enumerate(train_data):
        data = batch_data['data'] # batch_size * seq_len
        targets = batch_data['target'] # batch_size * seq_len
        words_lengths = batch_data['words_lengths']

        model.zero_grad()
        output = model(data, words_lengths) # output: batch_size * seq_len * vocab_size
        loss = criterion(output.view(-1, ntokens), targets.contiguous().view(targets.size(0) * targets.size(1)))
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.5f} | '
                    'loss {:5.5f} | ppl {:8.5f}'.format(
                epoch, i, len(train_data), lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.5f} | '
                  'loss {:5.5f} | ppl {:8.5f}'.format(
                epoch, i, len(train_data), lr,
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        logging.info('-' * 89)
        print('| end of epoch {:3d} | time: {:5.5f}s | valid loss {:5.5f} | '
                'valid ppl {:8.5f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        logging.info('| end of epoch {:3d} | time: {:5.5f}s | valid loss {:5.5f} | '
                'valid ppl {:8.5f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        logging.info('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
            if lr < 1e-5:
                break
except KeyboardInterrupt:
    print('-' * 89)
    logging.info('-' * 89)
    print('Exiting from training early')
    logging.info('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
logging.info('=' * 89)
print('| End of training | test loss {:5.5f} | test ppl {:8.5f}'.format(
    test_loss, math.exp(test_loss)))
logging.info('| End of training | test loss {:5.5f} | test ppl {:8.5f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
logging.info('=' * 89)