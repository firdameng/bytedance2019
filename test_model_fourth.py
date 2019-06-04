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


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'query_id': tf.FixedLenFeature([], tf.int64),
            'query': tf.FixedLenFeature([], tf.string),
            'query_title_id': tf.FixedLenFeature([], tf.int64),
            'title': tf.FixedLenFeature([], tf.string),
        })

    return features['query_id'], features['query'], features['query_title_id'], features['title']


def get_all_data(size=100000):
    query_id_list = []
    query_list = []
    query_title_id_list = []
    title_list = []
    tfrecords_list = [os.path.join(dirpath, filename) for dirpath, dirnames, filenames in os.walk(TFRECORDS_PATH) \
                      for filename in filenames]

    input_files = tf.placeholder(tf.string)
    dataset = tf.data.TFRecordDataset(input_files)
    dataset = dataset.map(parser)
    iterator = dataset.make_initializable_iterator()
    query_id, query, query_title_id, title = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer, feed_dict={input_files: tfrecords_list})
        count = 0
        while True:
            try:
                Query_Id, Query, Query_Title_Id, Title = sess.run(
                    [query_id, query, query_title_id, title])
                count += 1
                query_id_list.append(Query_Id)
                query_list.append(str(Query))
                query_title_id_list.append(Query_Title_Id)
                title_list.append(str(Title))
                # 只取10万条数据
                if count % (size) == 0:
                    yield np.array(query_id_list[int(count / size - 1) * size:]), np.array(
                        query_list[int(count / size - 1) * size:]), np.array(
                        query_title_id_list[int(count / size - 1) * size:]), np.array(
                        title_list[int(count / size - 1) * size:])
            except tf.errors.OutOfRangeError:
                break


if __name__ == '__main__':
    DATA_PATN = '/home/kesci/input/bytedance/first-round'
    WORK_PATH = '/home/kesci/work'
    TRAIN_MODEL_PATH = '/home/kesci/work/checkpoint_dir/'
    TRAIN_PATH = os.path.join(DATA_PATN, 'test.csv')
    TFRECORDS_PATH = os.path.join(WORK_PATH, 'tfrecords')

    with tf.Session() as session:
        print('读取训练模型')
        session.run(tf.global_variables_initializer())
        model_path = TRAIN_MODEL_PATH + 'model-dnn-0.meta'
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
        predictions = graph.get_tensor_by_name('output/Sigmoid:0')
        print('读取词汇表')
        vocab_processor = VocabularyProcessor.restore('/home/kesci/work/model/word_dict.pkl')

        data_iter = get_all_data(size=100000)
        for i in range(50):
            print('读取测试数据：', i)
            query_id, query, query_title_id, title = next(data_iter)
            print('词汇表转换:', i)
            x_query = np.array(list(vocab_processor.fit_transform(query)))
            x_title = np.array(list(vocab_processor.fit_transform(title)))

            feed_dict = {
                input_query: x_query,
                input_title: x_title,
                dropout_keep_prob: 0.7
            }
            predicts = session.run(predictions, feed_dict)
            print('开始写入文件:', i)
            with open('answer_test.csv', 'a') as f:
                csv_write = csv.writer(f)
                # 一行一行的写入csv文件
                for query_id, query_title_id, prediction in zip(query_id, query_title_id, predicts[:, 1]):
                    csv_write.writerow([query_id, query_title_id, prediction])