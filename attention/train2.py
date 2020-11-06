# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from attention_seq2seq import AttentionSeq2seq

# 載入資料
# x_train, t_train = sequence.load_data_without_test('ner_train.txt')
# x_test, t_test = sequence.load_data_without_test('ner_valid.txt')
(x_train, t_train), (x_test, t_test) = sequence.load_data('ner_valid.txt')
char_to_id, id_to_char = sequence.get_vocab()

# 反轉輸入內容
x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

# 設定超參數
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 256
batch_size = 128
max_epoch = 10
max_grad = 5.0

model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
# model = Seq2seq(vocab_size, wordvec_size, hidden_size)
# model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)

optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
    print("Training epoch %d" % (epoch,))
    trainer.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)

    total = len(x_test)
    print("Evaluating epoch %d, Total: %d" % (epoch, total))
    correct_num = 0
    for i in range(total):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i % 100 == 0
        if verbose:
            print("[%d]"%(i,), end = ' ')
        correct_num += eval_seq2seq(model, question, correct,
                                    id_to_char, verbose, is_reverse=True)
        # if i % 100 == 0:
        #     print("[%d]" % (i, ), end = " ")

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('\nval acc %.3f%%' % (acc * 100))


model.save_params()

# 繪圖
x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker='o')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(-0.05, 1.05)
plt.show()
