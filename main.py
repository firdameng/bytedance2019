import re
import os
import csv
import six
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import collections
from tensorflow.python.platform import gfile
from tensorflow.python.util.deprecation import deprecated


# 这就是改的源码的那两个类
class CategoricalVocabulary(object):

    @deprecated(None, 'Please use tensorflow/transform or tf.data.')
    def __init__(self, unknown_token="<UNK>", support_reverse=True):
        self._unknown_token = unknown_token
        self._mapping = {unknown_token: 0}
        self._support_reverse = support_reverse
        if support_reverse:
            self._reverse_mapping = [unknown_token]
        self._freq = collections.defaultdict(int)
        self._freeze = False

    def __len__(self):
        """Returns total count of mappings. Including unknown token."""
        return len(self._mapping)

    def freeze(self, freeze=True):
        self._freeze = freeze

    def get(self, category):
        if category not in self._mapping:
            if self._freeze:
                return 0
            self._mapping[category] = len(self._mapping)
            if self._support_reverse:
                self._reverse_mapping.append(category)
        return self._mapping[category]

    def add(self, category, count=1):
        """Adds count of the category to the frequency table.

        Args:
          category: string or integer, category to add frequency to.
          count: optional integer, how many to add.
        """
        category_id = self.get(category)
        if category_id <= 0:
            return
        if category not in self._freq:
            self._freq[category] = 1
        else:
            self._freq[category] += count

    def trim(self, min_frequency, max_frequency=-1):
        # Sort by alphabet then reversed frequency.
        self._freq = sorted(
            sorted(
                six.iteritems(self._freq),
                key=lambda x: (isinstance(x[0], str), x[0])),
            key=lambda x: x[1],
            reverse=True)
        self._mapping = {self._unknown_token: 0}
        if self._support_reverse:
            self._reverse_mapping = [self._unknown_token]
        idx = 1
        for category, count in self._freq:
            if max_frequency > 0 and count >= max_frequency:
                continue
            if count <= min_frequency:
                break
            self._mapping[category] = idx
            idx += 1
            if self._support_reverse:
                self._reverse_mapping.append(category)
        self._freq = dict(self._freq[:idx - 1])

    def reverse(self, class_id):
        if not self._support_reverse:
            raise ValueError("This vocabulary wasn't initialized with "
                             "support_reverse to support reverse() function.")
        return self._reverse_mapping[class_id]


try:
    import cPickle as pickle
except ImportError:
    import pickle

TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",
                          re.UNICODE)


@deprecated(None, 'Please use tensorflow/transform or tf.data.')
def tokenizer(iterator):
    for value in iterator:
        yield TOKENIZER_RE.findall(value)


@deprecated(None, 'Please use tensorflow/transform or tf.data.')
class ByteProcessor(object):

    @deprecated(None, 'Please use tensorflow/transform or tf.data.')
    def __init__(self, max_document_length):
        self.max_document_length = max_document_length

    def fit(self, x):
        """Does nothing. No fitting required."""
        pass

    def fit_transform(self, x):
        """Calls transform."""
        return self.transform(x)

    # pylint: disable=no-self-use
    def reverse(self, x):
        for data in x:
            document = np.trim_zeros(data.astype(np.int8), trim='b').tostring()
            try:
                yield document.decode('utf-8')
            except UnicodeDecodeError:
                yield ''

    def transform(self, x):
        if six.PY3:
            # For Python3 defined buffer as memoryview.
            buffer_or_memoryview = memoryview
        else:
            buffer_or_memoryview = buffer  # pylint: disable=undefined-variable
        for document in x:
            if isinstance(document, six.text_type):
                document = document.encode('utf-8')
            document_mv = buffer_or_memoryview(document)
            buff = np.frombuffer(document_mv[:self.max_document_length],
                                 dtype=np.uint8)
            yield np.pad(buff, (0, self.max_document_length - len(buff)), 'constant')


class VocabularyProcessor(object):

    @deprecated(None, 'Please use tensorflow/transform or tf.data.')
    def __init__(self,
                 max_document_length,
                 min_frequency=0,
                 vocabulary=None,
                 tokenizer_fn=None):
        self.max_document_length = max_document_length
        self.min_frequency = min_frequency
        if vocabulary:
            self.vocabulary_ = vocabulary
        else:
            self.vocabulary_ = CategoricalVocabulary()
        if tokenizer_fn:
            self._tokenizer = tokenizer_fn
        else:
            self._tokenizer = tokenizer

    def fit(self, raw_documents, unused_y=None):
        for tokens in self._tokenizer(raw_documents):
            for token in tokens:
                self.vocabulary_.add(token)
        if self.min_frequency > 0:
            self.vocabulary_.trim(self.min_frequency)
        self.vocabulary_.freeze()
        return self

    def fit_transform(self, raw_documents, unused_y=None):
        self.fit(raw_documents)
        return self.transform(raw_documents)

    def transform(self, raw_documents):
        for tokens in self._tokenizer(raw_documents):
            word_ids = np.zeros(self.max_document_length, np.int64)
            for idx, token in enumerate(tokens):
                if idx >= self.max_document_length:
                    break
                word_ids[idx] = self.vocabulary_.get(token)
            yield word_ids

    def reverse(self, documents):
        for item in documents:
            output = []
            for class_id in item:
                output.append(self.vocabulary_.reverse(class_id))
            yield ' '.join(output)

    def save(self, filename):
        with gfile.Open(filename, 'wb') as f:
            f.write(pickle.dumps(self))

    @classmethod
    def restore(cls, filename):
        with gfile.Open(filename, 'rb') as f:
            return pickle.loads(f.read())


# 这就是最普通的获取训练数据
# '../input/bytedance/first-round/train.csv'
def get_data(size=100, path='./data/train.csv'):
    query_id = []
    query = []
    query_title_id = []
    title = []
    label = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        count = 0
        for row in csv_reader:
            count += 1
            query_id.append(row[0])
            query.append(row[1])
            query_title_id.append(row[2])
            title.append(row[3])
            temp_label = int(row[4].strip())
            if temp_label == 1:
                label.append([0, 1])
            elif temp_label == 0:
                label.append([1, 0])
            # 只取size条数据
            if count % size == 0:
                yield np.array(query_id[int(count / size - 1) * size:]), \
                      np.array(query[int(count / size - 1) * size:]), \
                      np.array(query_title_id[int(count / size - 1) * size:]), \
                      np.array(title[int(count / size - 1) * size:]), \
                      np.array(label[int(count / size - 1) * size:])

# '../input/bytedance/first-round/test.csv'
def get_test_data(size=100000, path='./data/test.csv'):
    query_id = []
    query = []
    query_title_id = []
    title = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        count = 0
        for row in csv_reader:
            count += 1
            query_id.append(row[0])
            query.append(row[1])
            query_title_id.append(row[2])
            title.append(row[3])
            # 只取size条数据
            if count % size == 0:
                yield np.array(query_id[int(count / size - 1) * size:]), \
                      np.array(query[int(count / size - 1) * size:]), \
                      np.array(query_title_id[int(count / size - 1) * size:]), \
                      np.array(title[int(count / size - 1) * size:])


def get_sentence_length(x_text, per=95):
    lens = [len(x.split()) for x in x_text]
    sentence_length = sorted(lens)[int(len(lens) * per / 100.0)]
    return sentence_length


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    # 最多重复取多少次数据
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        # 每一次获取batch_size个训练数据
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# DNN模型
class DNN:

    # 模型中的变量尽量都设置一个name，不然最后读取模型时不方便获取到变量
    def __init__(self, sentence_length, num_classes, vocab_size, embedding_size, depth, l2):
        l2_loss = tf.constant(0.0)
        regularization = tf.contrib.layers.l2_regularizer(l2)
        self.input_query = tf.placeholder(tf.int32, [None, sentence_length], name='query')
        self.input_title = tf.placeholder(tf.int32, [None, sentence_length], name='title')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='label')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("embedding"):
            self.word_embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="word")
            self.title = tf.nn.embedding_lookup(self.word_embedding, self.input_title, name='title')
            # W的shape为（n, sentence_length, embedding_size）
            self.W = tf.nn.embedding_lookup(self.word_embedding, self.input_query, name='w')
            self.b = tf.Variable(tf.constant(0.1, shape=[sentence_length, embedding_size]), name="b")

        # DNN添加多个全连阶层
        self.pre_layer = tf.layers.dense(self.W, embedding_size, tf.nn.relu) + self.b
        for deep in depth:
            name = 'biases-' + str(deep)
            biases = tf.Variable(tf.constant(0.1, shape=[deep]), name=name)
            # shape(n, sentence_length, deep)
            layer = tf.layers.dense(self.pre_layer, deep, tf.nn.relu) + biases
            self.pre_layer = layer

        # drop层,防止过拟合,参数为dropout_keep_prob
        with tf.name_scope("dropout"):
            # shape(n, sentence_length, deep)
            self.drop = tf.nn.dropout(self.pre_layer, self.dropout_keep_prob, name='drop')

        # 最终（非标准化）分数和预测
        with tf.name_scope("output"):
            biases = tf.Variable(tf.constant(0.1, shape=[embedding_size]), name="biases")
            # shape(n, embedding_size)
            self.final_layer = tf.reduce_sum(
                tf.subtract(tf.layers.dense(self.drop, embedding_size) + biases, self.title), 1)
            # shape(n, num_classes)
            self.result_biases = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="result_biases")
            self.result_layer = tf.layers.dense(self.final_layer, num_classes) + self.result_biases
            # 调用tf.nn.softmax方法，方法结果是预测概率值；
            # if num_classes == 2:
            #     self.predictions = tf.nn.sigmoid(self.result_layer, name='predictions')
            # else:
            self.predictions = tf.nn.softmax(self.result_layer, name='predictions')
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y, logits=self.result_layer)

        # 计算平均交叉熵损失
        with tf.name_scope("loss"):
            # l2_loss 计算公式：output = sum(t**2)/2
            l2_loss += regularization(cross_entropy)
            self.loss = tf.reduce_mean(cross_entropy, name='loss') + l2_loss

        # 正确率
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


def train(session, global_step, train_op, model_name, x_train_query, x_train_title, y_train_label, x_test_query,
          x_test_title, y_test_label, batch_size, dropout, l2, model):
    # 用于保存模型的图结构
    saver = tf.train.Saver()

    def train_step(x_batch_query, x_batch_title, y_batch):
        # 填充词典
        feed_dict = {
            model.input_query: x_batch_query,
            model.input_title: x_batch_title,
            model.input_y: y_batch,
            model.dropout_keep_prob: dropout
        }
        _, step, loss, accuracy, drop = session.run([train_op, global_step, model.loss, model.accuracy, model.drop],
                                                    feed_dict)
        return accuracy

    def test_step(x_batch_query, x_batch_title, y_batch):
        # 填充词典
        feed_dict = {
            model.input_query: x_batch_query,
            model.input_title: x_batch_title,
            model.input_y: y_batch,
            model.dropout_keep_prob: dropout
        }
        step, loss, accuracy = session.run(
            [global_step, model.loss, model.accuracy], feed_dict)
        print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
        return accuracy, loss

    # 生成一批次数据
    train_accuracies = []
    test_accuracies = []
    times = 0
    max_acc = 0
    pre_acc = 0
    batches = batch_iter(list(zip(x_train_query, x_train_title, y_train_label)), batch_size,
                         batch_times)
    # 保存模型的图结构
    saver.save(session, model_name)
    for batch in batches:
        x_batch_query, x_batch_title, y_batch = zip(*batch)
        train_accuracy = train_step(x_batch_query, x_batch_title, y_batch)
        # 如果拟合完成，那么就停止运算
        if train_accuracy > 0.99:
            times += 1
            if times > 350:
                break
        current_step = tf.train.global_step(session, global_step)
        # 每200步做一次记录
        if current_step % 200 == 0:
            print('\n进行检查点测试测试集:\n')
            train_accuracies.append(train_accuracy)
            print(train_accuracy)
            temp, loss = test_step(x_test_query, x_test_title, y_test_label)
            if temp > max_acc:
                max_acc = temp
                # 保存模型，write_meta_graph设为模型表示只保存模型中的数据，而不需要再次保存图的结构
                # 这个保存必须是最后一个保存，因为读取模型的时候，就是读取的最后一个模型
                # 如果最后一个模型不是最好的数据，效果肯定差
                saver.save(session, model_name, write_meta_graph=False)
            test_accuracies.append(temp)
            if abs(pre_acc - temp) < 0.0015:
                times += 1
                if times >= 5:
                    break
            pre_acc = temp

    return train_accuracies, test_accuracies


def get_answer():
    with tf.Session() as session:
        print('读取训练模型')
        session.run(tf.global_variables_initializer())
        model_path = TRAIN_MODEL_PATH + 'model-dnn.meta'
        # 读取模型
        saver = tf.train.import_meta_graph(model_path)
        # 读取模型最后一个检查点数据，就是最好的那个数据
        saver.restore(session, tf.train.latest_checkpoint(TRAIN_MODEL_PATH))

        # graph请放在这里，放到前面就会报错
        graph = tf.get_default_graph()
        input_query = graph.get_tensor_by_name('query:0')
        input_title = graph.get_tensor_by_name('title:0')
        dropout_keep_prob = graph.get_tensor_by_name('dropout_keep_prob:0')
        accuracy = graph.get_tensor_by_name('accuracy/accuracy:0')
        # 因为之前这个变量没有设置名字，所以我只有使用默认名字获取变量，所以一定要设置名字
        predictions = graph.get_tensor_by_name('output/predictions:0')
        print('读取词汇表')
        vocab_processor = VocabularyProcessor.restore(word_dict_name)
        # 这一句话必须要
        vocab_processor.vocabulary_.freeze(freeze=True)
        data_iter = get_test_data()
        for i in range(50):
            print('读取测试数据：', i)
            query_id, query, query_title_id, title = next(data_iter)
            print('词汇表转换:', i)
            x_query = np.array(list(vocab_processor.transform(query)))
            x_title = np.array(list(vocab_processor.transform(title)))

            feed_dict = {
                input_query: x_query,
                input_title: x_title,
                dropout_keep_prob: 0.6
            }
            predicts = session.run(predictions, feed_dict)
            print('开始写入文件:', i)
            with open('answer_test.csv', 'a') as f:
                csv_write = csv.writer(f)
                # 一行一行的写入csv文件
                for query_id, query_title_id, prediction in zip(query_id, query_title_id, predicts[:, 1]):
                    csv_write.writerow([query_id, query_title_id, prediction])


def main(args=None):
    np.random.seed(3)
    # 设置取随机的参数构成模型
    random_para_index = list(np.random.permutation(np.arange(5)))
    embedding_size = embedding_sizes[random_para_index[0]]
    depth = depths[random_para_index[1]]
    l2 = l2s[random_para_index[2]]
    dropout = dropouts[random_para_index[3]]
    batch_size = batch_sizes[random_para_index[4]]

    with tf.Graph().as_default():
        # 获取保存好的词汇表
        vocab_processor = VocabularyProcessor.restore(word_dict_name)
        vocab_size = len(vocab_processor.vocabulary_)
        # 构建dnn神经网络
        dnn = DNN(sentence_length, num_classes, vocab_size, embedding_size, depth, l2)
        # 进行session参数配置
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session = tf.Session(config=session_conf)
        with session.as_default():
            # 定义训练程序
            global_step = tf.Variable(0, name="global_step", trainable=False)
            # 优化器
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(dnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            # 初始化所有变量
            session.run(tf.global_variables_initializer())

        # 训练数据迭代器
        data_iter = get_data(size=1000000)
        for i in range(model_num):
            query_id, query, query_title_id, title, label = next(data_iter)
            # 构建嵌入层词向量
            x_query = np.array(list(vocab_processor.fit_transform(query)))
            x_title = np.array(list(vocab_processor.fit_transform(title)))
            # 打乱训练数据
            np.random.seed(i)

            # 输出所用参数
            print(i, embedding_size, depth, l2, dropout, batch_size)

            shuffle_indices = np.random.permutation(np.arange(len(label)))
            x_query_shuffled = x_query[shuffle_indices]
            x_title_shuffled = x_title[shuffle_indices]
            y_shuffled = label[shuffle_indices]
            # 打乱顺序后得到训练集和测试集
            x_train_query = x_query_shuffled[int(test_size * len(label)):]
            x_train_title = x_title_shuffled[int(test_size * len(label)):]
            y_train_label = y_shuffled[int(test_size * len(label)):]
            x_test_query = x_query_shuffled[0: int(test_size * len(label))]
            x_test_title = x_title_shuffled[0: int(test_size * len(label))]
            y_test_label = y_shuffled[0: int(test_size * len(label))]

            # 删除没用的
            del x_query_shuffled, x_title_shuffled, y_shuffled
            model_name = TRAIN_MODEL_PATH + 'model-dnn'
            # 构建dnn神经网络
            # dnn = DNN(sentence_length, num_classes, vocab_size, embedding_size, depth, l2)
            # 训练
            train_accuracies, test_accuracies = train(session, global_step, train_op, model_name, x_train_query,
                                                      x_train_title, y_train_label, x_test_query,
                                                      x_test_title, y_test_label, batch_size, dropout, l2, dnn)
            title = str(i) + ' | ' + str(embedding_size) + ' | ' + str(l2) + ' | ' + str(
                batch_size) + ' | ' + str(dropout)
            # 保存记录
            with open('test_acc.txt', 'a') as file:
                if i % 5 == 0:
                    file.write('\n')
                file.write(title + ' | ' + str(max(test_accuracies)) + ' | ' + '\n')
            # 绘制正确率图
            plt.plot(train_accuracies, c='r')
            plt.plot(test_accuracies, c='b')
            plt.grid(True)
            plt.title(title)
            plt.show()


def get_word_dict():
    # 这个是获得已经保存好的词典
    vocab_processor = VocabularyProcessor(sentence_length, 5)
    # 就是因为那个方法的文件报错了，所以我更改了索引，跳过了报错文件
    data_iter = get_data()
    pre_words = 0
    times = 0
    for i in range(1000):
        query_id, query, query_title_id, title, label = next(data_iter)
        all_sentence = np.concatenate([query, title], axis=0)
        # 调用fit方法，提取词典
        vocab_processor.fit(all_sentence)
        # 设置为false，源码中，每次fit后都会设为true，导致无法分段提取词典
        # 所以这个false是必须的
        vocab_processor.vocabulary_.freeze(freeze=False)
        len_diff = len(vocab_processor.vocabulary_) - pre_words
        print('词典提取，第', (i + 1) * 10, '万条数据===========提取单词数', len(vocab_processor.vocabulary_), ' | ', len_diff)
        pre_words = len(vocab_processor.vocabulary_)
        # 保存提取出来的词典
        vocab_processor.save(word_dict_name)
        # 如果已经难以提取出词，那么就结束
        if len_diff < 100:
            times += 1
            if times > 8:
                break
        del query_id, query, query_title_id, title, label, all_sentence
    print(len(vocab_processor.vocabulary_))
    vocab_processor.save(word_dict_name)


if __name__ == "__main__":
    # 设置基本参数
    DATA_PATN = './data'
    WORK_PATH = './'
    TRAIN_MODEL_PATH = './checkpoint_dir/'
    TRAIN_PATH = os.path.join(DATA_PATN, 'train.csv')
    TFRECORDS_PATH = os.path.join(WORK_PATH, 'tfrecords')
    word_dict_name = './model/word-dict-16-5.pkl'
    sentence_length = 16
    # 得到词典
    print('得到词典')
    # get_word_dict()
    # print('词典构造完成')
    # 设置进行训练的参数
    test_size = 0.05
    num_classes = 2
    model_num = 80

    embedding_sizes = [96, 128, 175, 216, 275]
    depths = [[20, 15, 10, 5, 3], [18, 15, 12, 9, 6, 3], [25, 20, 15, 10, 5], [14, 12, 10, 8, 6, 4], [20, 15, 10, 5]]
    l2s = [0, 0.001, 0.002, 0.005, 0.01]
    dropouts = [0.7, 0.75, 0.8, 0.85, 0.9]
    batch_sizes = [320, 480, 640, 800, 960]
    batch_times = 10
    print('构建模型')
    main()
    print('模型训练完成')
    # print('根据模型，得到答案')
    # print(len(VocabularyProcessor.restore(word_dict_name).vocabulary_))
    # get_answer()
    # print('答案已保存，祝自己好运')
