#! -*- coding: utf-8 -*-
# 百度关系抽取pipline模型NER

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.backend import multilabel_categorical_crossentropy
from bert4keras.layers import GlobalPointer
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from keras.models import Model
from tqdm import tqdm
import fairies as fa
from bert4keras.backend import keras, search_layer, K

maxlen = 256
epochs = 20
batch_size = 16
learning_rate = 1e-5
categories = set()

def load_data(filename):
    
    """加载数据
        单条格式：[text, (start, end, label), (start, end, label), ...]，
                  意味着text[start:end + 1]是类型为label的实体。
    """
    
    D = []
    data = fa.read_json(filename)

    for d in data:
        text = d['text']
        spo_list = d['spo_list']

        entry_list = []
        entry_list.append(text)

        for s in spo_list:
             
            predicate = s['predicate']
            subject_type = predicate + 's_' + s['subject_type']
            categories.add(subject_type)
            subject = s['subject']
            if text.find(subject) != -1:
                entry_list.append([text.find(subject),text.find(subject)+len(subject)-1,subject_type])

            for ob in s['object']:
                ob_ject_type = predicate + 'o_' + s['object_type'][ob]
                categories.add(ob_ject_type)
                ob_ject = s['object'][ob]
                if text.find(ob_ject) != -1:
                    entry_list.append([text.find(ob_ject),text.find(ob_ject)+len(ob_ject)-1,ob_ject_type])
        
        D.append(entry_list)

    return D

# 标注数据
train_data = load_data('duie_train.json')
valid_data = load_data('duie_dev.json')

# test_data = load_data('duie_test.json')
categories = list(sorted(categories))
print(categories)
print(len(categories))

# 建立分词器
config_path = 'chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):

            tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
            mapping = tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}

            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            labels = np.zeros((len(categories), maxlen, maxlen))

            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    label = categories.index(label)
                    labels[label, start, end] = 1
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels[:, :len(token_ids), :len(token_ids)])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels, seq_dims=3)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def global_pointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    bh = K.prod(K.shape(y_pred)[:2])
    y_true = K.reshape(y_true, (bh, -1))
    y_pred = K.reshape(y_pred, (bh, -1))
    return K.mean(multilabel_categorical_crossentropy(y_true, y_pred))

def global_pointer_f1_score(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
    y_pred = K.cast(K.greater(y_pred, 0), K.floatx())
    return 2 * K.sum(y_true * y_pred) / K.sum(y_true + y_pred)

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


model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
) 

output = GlobalPointer(len(categories), 64)(model.output)

model = Model(model.input, output)
model.summary()

model.compile(
    loss=global_pointer_crossentropy,
    optimizer=Adam(learning_rate),
    metrics=[global_pointer_f1_score]
)

class NamedEntityRecognizer(object):
    """命名实体识别器
    """
    def recognize(self, text, threshold=0):
        tokens = tokenizer.tokenize(text, maxlen=512)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        scores = model.predict([token_ids, segment_ids])[0]
        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf
        entities = []
        for l, start, end in zip(*np.where(scores > threshold)):
            entities.append(
                (mapping[start][0], mapping[end][-1], categories[l])
            )
        return entities

NER = NamedEntityRecognizer()

def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        R = set(NER.recognize(d[0]))
        T = set([tuple(i) for i in d[1:]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):

        model.save_weights('last_model.weights')
        f1, precision, recall = evaluate(valid_data)
        
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('best_model.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
       

if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)
    
    adversarial_training(model, 'Embedding-Token', 0.5)
    
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )
        
    model.load_weights('81_best_model.weights')

    test_data = fa.read_json('duie_test2.json')
        
    entry_new_res_512 = []
    for t in tqdm(test_data, ncols=100):
#     for t in test_data:
        text = t['text']
        R = set(NER.recognize(text))
        entry_list = []
        for i in R:
            entry_list.append([text[i[0]:i[1]+1],i[2]])
#         print(entry_list)
        entry_new_res_512.append([text,entry_list])
        
    fa.write_json('81_entry_duie_test2.json',entry_new_res_512)
