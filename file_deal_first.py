import tensorflow as tf
import os
import pandas as pd

# 这个代码使用来转换文件的，将他给的原始的文件，分割成1000个每个10万条数据的二进制文件
# 当然也可以用来分割测试数据，用这个要注意，一亿条的训练数据要一晚上
# 而且这个保存的文件似乎读取时会出问题，至少我这边遇到问题了
# 很坑人，运行到一半发现出问题，所以不时很建议这个
DATA_PATN = '/home/kesci/input/bytedance/first-round'
WORK_PATH = '/home/kesci/work'
# test.csv是测试数据， train.csv是一亿条训练数据
TRAIN_PATH = os.path.join(DATA_PATN, 'test.csv')
TFRECORDS_PATH = os.path.join(WORK_PATH, 'tfrecords')
if not os.path.exists(TFRECORDS_PATH):
    os.mkdir(TFRECORDS_PATH)

NUM_SHARDS = 50   # 总共写入多少文件（1000个）
INSTANCES_PER_SHARD = 100000    # 每个文件写入多少数据（10万）

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

# 这个是一个读取csv的对象，每次读取INSTANCES_PER_SHARD次
train = pd.read_csv(TRAIN_PATH, header=None, chunksize=INSTANCES_PER_SHARD)

# 保存数据
for index, sub_train in enumerate(train):
    TFRECORDS_TRAIN = os.path.join(TFRECORDS_PATH, 'test.tfrecords-%.4d-of-%.4d'%(index, NUM_SHARDS))
    writer = tf.python_io.TFRecordWriter(TFRECORDS_TRAIN)
    print(index)
    for item in sub_train.iterrows():
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'query_id': _int64_feature(item[1][0]),
                    'query': _bytes_feature(item[1][1].encode()),
                    'query_title_id': _int64_feature(item[1][2]),
                    'title': _bytes_feature(item[1][3].encode()),
                    'label': _int64_feature(item[1][4])
                }
            )
        )
        writer.write(example.SerializeToString())
    writer.close()

# 下面注释的是读取文件
# def parser(record):

#     features = tf.parse_single_example(
#         record,
#         features={
#             'query_id': tf.FixedLenFeature([], tf.int64),
#             'query' : tf.FixedLenFeature([], tf.string),
#             'query_title_id': tf.FixedLenFeature([], tf.int64),
#             'title' : tf.FixedLenFeature([], tf.string),
#             'label': tf.FixedLenFeature([], tf.int64),
#       })

#     return features['query_id'], features['query'], features['query_title_id'], features['title'], features['label']

# tfrecords_list = [os.path.join(dirpath, filename) for dirpath, dirnames, filenames in os.walk(TFRECORDS_PATH) \
#                                                     for filename in filenames]

# input_files = tf.placeholder(tf.string)
# dataset = tf.data.TFRecordDataset(input_files)
# dataset = dataset.map(parser)
# iterator = dataset.make_initializable_iterator()
# query_id, query, query_title_id, title, label = iterator.get_next()

# with tf.Session() as sess:
#     sess.run(iterator.initializer,feed_dict={input_files: tfrecords_list})
#     while True:
#         try:
#             Query_Id, Query, Query_Title_Id, Title, Label = sess.run([query_id, query, query_title_id, title, label])
#         except tf.errors.OutOfRangeError:
#             break