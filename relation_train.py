import json
import numpy as np
from bert4keras.backend import keras, set_gelu, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
import fairies as fa
from keras.layers import *
import random 
from bert4keras.backend import keras, search_layer, K


a =[
    "注册资本",
    "作者",
    "所属专辑",
    "歌手",
    "邮政编码",
    "主演",
    "上映时间",
    "上映时间",
    "饰演",
    "饰演",
    "国籍",
    "成立日期",
    "毕业院校",
    "作曲",
    "作词",
    "编剧",
    "导演",
    "面积",
    "占地面积",
    "总部地点",
    "制片人",
    "嘉宾",
    "简称",
    "主持人",
    "获奖",
    "获奖",
    "获奖",
    "获奖",
    "海拔",
    "出品公司",
    "配音",
    "配音",
    "所在城市",
    "号",
    "主角",
    "创始人",
    "父亲",
    "祖籍",
    "母亲",
    "朝代",
    "董事长",
    "人口数量",
    "妻子",
    "丈夫",
    "票房",
    "票房",
    "专业代码",
    "气候",
    "修业年限",
    "改编自",
    "官方语言",
    "首都",
    "主题曲",
    "校长",
    "代言人"
]

b = [
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "inArea",
    "@value",
    "inWork",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "inWork",
    "onDate",
    "period",
    "@value",
    "@value",
    "@value",
    "inWork",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "inArea",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value",
    "@value"
]

c = []

c.append('无')
for i in range(len(a)):
    c.append(a[i] + '_' + b[i])

id2label,label2id = fa.label2id(c)
label_nums = len(c)

maxlen = 256
batch_size = 16

def read_data(filename):

    train_data = fa.read_json(filename)

    res = []
    for i in train_data:
        text = i['text']

        pos = []
        neg = []

        # with type
        pos_list = []

        subjectList = []
        objectList = []

        for spo in i['spo_list']:
            for obj in spo['object_type']:
                                   
                # "subject,object"
                # add predicate
                predicate = spo['predicate'] + '_' + obj
                pos_list.append([spo['subject'],spo['object'][obj],label2id[predicate],spo['subject_type'],spo['object_type'][obj]])

                pos.append([spo['subject'],spo['object'][obj],spo['subject_type'],spo['object_type'][obj]])
                if [spo['subject'],spo['subject_type']] not in subjectList:
                    subjectList.append([spo['subject'],spo['subject_type']])
                if [spo['object'][obj],spo['object_type'][obj]] not in objectList:
                    objectList.append([spo['object'][obj],spo['object_type'][obj]])

        for i in subjectList:
            for j in objectList:
                if [i[0],j[0],i[1],j[1]] not in pos:
                    neg.append([i[0],j[0],i[1],j[1]])

        random.shuffle(neg)

        neg = neg[:5]

        for i in pos_list:
            res.append([text,[i[0],i[1],i[3],i[4]],i[2]])
        for i in neg:
            res.append([text,i,0])
        
    return res        

a = read_data('duie_train.json')
dev_data = read_data('duie_dev.json')


p = 'D:/lyh/model/chinese_roberta_wwm_ext_L-12_H-768_A-12/'
config_path = p +'bert_config.json'
checkpoint_path = p + 'bert_model.ckpt'
dict_path = p +'vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (model._feed_inputs +
              model._feed_targets +
              model._feed_sample_weights)  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数

def search(pattern, sequence):

    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """

    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

class data_generator(DataGenerator):

    """数据生成器

    """

    def __iter__(self, random=False):

        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, result in self.sample(random):

            text = result[0]
            predicts = result[1]

            text_type = predicts[2] + '#' +  predicts[0] + '#' + predicts[3] + '#' +  predicts[1]

            token_ids, segment_ids = tokenizer.encode(text_type,text, maxlen=maxlen)
    
            labels = [result[2]]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)

num_labels = label_nums

output = Lambda(lambda x: x[:, 0],
                name='CLS-token')(bert.model.output)

# output = Dense(units=150,
#                 activation='relu',
#                 kernel_initializer=bert.initializer)(output)

# output = Dense(units=150,
#                 activation='relu',
#                 kernel_initializer=bert.initializer)(output)

output = Dense(units=num_labels,
               activation='softmax',
               kernel_initializer=bert.initializer)(output)

model = Model(bert.model.input, output)              

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(1e-5),
              metrics=['accuracy'])

train_generator = data_generator(a, batch_size)
dev_generator = data_generator(dev_data, batch_size)

def search_distance(pattern, sequence):

    res = []
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            res.append(i)
    return res

def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    print(right/total)    
    return right/total

def predict(data):
    res = []
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        new = y_pred.tolist()
        res.extend(new)
    return res   

class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        model.save_weights('last_model.weights')
        val_acc = evaluate(dev_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        print(val_acc)
        print(self.best_val_acc)
        fa.print_to_log(self.best_val_acc)

        fa.print_to_log(val_acc)
        fa.print_to_log(self.best_val_acc)
    
evaluator = Evaluator()

adversarial_training(model,'Embedding-Token',0.5)

model.summary()

model.fit_generator(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=20,
    callbacks=[evaluator]
)



