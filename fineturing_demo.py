'''
每天一句：
加油努力！当当当！
tensorflow-gpu的调用内存问题：
import tensorflow as tf
from keras import backend as K
config=tf.Configproto()
config.gpu_options.allow_growth=True
sess-tf.Session(config=config)
k.set_session(sess)
'''
"""
修改提供的demo使用今天学习的模型结构如rnn完成模型训练。
"""
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现
"""
class TorchModel(nn.Module):
    def __init__(self,input_dim,sentence_len,vocab):
        super(TorchModel, self).__init__()
        #embedding层输入向量：len(vocab)字符集字符总数，有多少个向量，input_dim向量化的维度
        #embedding层输出向量：输入x的对应向量
        self.embedding = nn.Embedding(len(vocab),input_dim)
        #加入rnn层数
        self.rnn = nn.RNN(input_size=input_dim,
                            hidden_size=input_dim,
                            batch_first=True)
        #参数含义：输入向量的维度|hidden_size希望rnn输出的维度|batch_first=ture表示batch_size是输入的第一维
        #rnn会返回【输出，隐藏状态】
        #加入双向RNN层数,输出为隐藏层在各个时间步上计算并输出的隐藏状态，形状为(时间步数, 批量大小, 隐藏单元个数)
        #隐藏状态指的是隐藏层在最后时间步的隐藏状态：当隐藏层有多层时，每⼀层的隐藏状态都会记录在该变量中
        self.classify = nn.Linear(input_dim,3)#线性层数:输入的维度为input_dim,输出维度为3
        self.loss = nn.functional.cross_entropy

    def forward(self,x,y=None):#输入x，真实值y、若传入真实值，则返回loss，若不传入y则返回预测结果
        #print(x.shape)#torch.Size([30, 6])
        x = self.embedding(x)     #(batch_size,sen_len)->(batch_size,sen_len,input_dim)
        #print(x.shape)#torch.Size([30, 6, 20])
        _,x = self.rnn(x)        #(batch_size,sen_len,input_dim)->(1,batch_size,input_dim)
        #print(x.shape)#torch.Size([2, 30, 20])
        #我们只需要拿到网络中最后一个时间步的隐藏状态
        y_pred = self.classify(x.squeeze())# (batch_size, input_dim) -> (batch_size, 3)
        #x.squeeze()去掉tensor中维数为1的维度 (1,batch_size,input_dim)->(batch_size,input_dim)

        #判断y的输出
        if y is not None:
            return self.loss(y_pred,y.squeeze())#输出loss
        else:
            return y_pred #输出预测结果

def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    vocab = {}
    for index,chars in enumerate(chars):
        vocab[chars] = index
    vocab['unk']=len(vocab)
    return vocab

def build_sample(vocab,sen_len):
    #随机从字表中选取sen_len个字形成一个文本
    x = [random.choice(list(vocab.keys())) for _ in range(sen_len)]
    #样本三分类
    if set("abcds") & set(x):
        y = 0
    elif set("zxv") &set(x):
        y = 1
    else:
        y = 2
    x = [vocab.get(word,vocab['unk']) for word in x]#将字转化成序号，做embedding
    return x,y

#建立数据集
def build_dataset(sample_len,vocab,sen_len):
    dataset_x = []
    dataset_y = []
    for i in range(sample_len):
        x,y = build_sample(vocab,sen_len)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x),torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab,char_dim,sen_len):
    model = TorchModel(input_dim=char_dim,sentence_len=sen_len,vocab=vocab)
    return model
#测试代码
#测试每轮训练模型的准确率
def evaluate_model(model,vocab,sample_len):
    model.eval()
    #建立200个用于测试的样本,样本文本长度为sample_len
    x,y = build_dataset(sample_len=200,vocab=vocab,sen_len=sample_len)
    print("本次预测集中共有%d个0类，%d个1类，%d个二类样本"%(sum(y.eq(0)), sum(y.eq(1)), sum(y.eq(2))))
    correct,wrong = 0,0
    with torch.no_grad():
        y_pred = model(x)
        y_pred = torch.argmax(y_pred,dim=-1)
        correct +=int(sum(y_pred==y.squeeze()))
        wrong += len(y)-correct
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    #训练参数配置
    epoch_num = 20  #训练次数
    batch_size = 30 #每轮训练的样本个数
    char_dim = 20   #每个字的维度
    train_sample = 500    #每轮训练总共训练的样本总数
    sen_len = 8     #样本的文本长度
    lr = 0.005      #学习率

    #构建训练字表
    vocab = build_vocab()

    #建立模型
    model = build_model(vocab=vocab,char_dim=char_dim,sen_len=sen_len)

    #选择优化器
    optim = torch.optim.Adam(model.parameters(),lr=lr)
    log = []

    #训练模型
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample/batch_size)):#一共有train_sample/batch_size种样本
            #构造训练样本
            x,y = build_dataset(batch_size,vocab,sen_len)#抽取batch_size个样本
            optim.zero_grad()#梯度归零
            loss = model(x,y)#模型y输入有，输出loss
            loss.backward()#计算梯度
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate_model(model,vocab,sample_len=sen_len)
        log.append([acc,np.mean(watch_loss)])

    #保存模型
    torch.save(model.state_dict(), "model_save.pth")
    #保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

def predict(model_path,vocab_path,input_strings):
    char_dim = 20
    sen_len = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))#加载字符表
    model = build_model(vocab=vocab,char_dim=char_dim,sen_len=sen_len)#建立模型
    model.load_state_dict(torch.load(model_path))#加载模型参数
    x = []
    for input_string in input_strings:
        #输入序列化
        x.append([vocab.get(char,vocab["unk"]) for char in input_string])

    print(x)
    model.eval()
    with torch.no_grad():#不计算梯度
        result = model.forward(torch.LongTensor(x))
    print(result)
    for i,input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%s"%(input_string, int(torch.argmax(result[i])), result[i])) #打印结果


if __name__=="__main__":
    #main()
    test_strings = ["abcdse", "123456", "234sdj", "jdheyn"]
    predict("model_save.pth", "vocab.json", test_strings)














