"""convert html json files into TFRecords files."""
import concurrent.futures
import nltk
from urllib.parse import urlparse
import sqlite3
from io import StringIO
import time
import logging
import json
import os
import math
from random import shuffle

import numpy as np
import tensorflow as tf

#########################################
# FLAGS
#########################################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("data_dir", "data/", 'path of data directory.')
tf.app.flags.DEFINE_integer("num_cats", 5,
                            "the number of categories of dataset.")
tf.app.flags.DEFINE_string("dataset_type", "ukwa", 'ukwa, dmoz, mcafee')
# tf.app.flags.DEFINE_string(
#     "categories", "Arts,Business,Computers,Health,Sports",
#     'categories name list, divided by comma, no space in between.')
# tf.app.flags.DEFINE_string(
#     "categories",
#     "Arts,Business,Computers,Health,Sports,Society,Science,Recreation,"
#     "Shopping,Reference,Games,Home",
#     'categories name list, divided by comma, no space in between.')
tf.app.flags.DEFINE_integer(
    "we_dim", 50,
    "number of characters in each input sequences (default: 1024)")
tf.app.flags.DEFINE_string("html_folder", "html-time/",
                           'path of html json files directory under data_dir.')
tf.app.flags.DEFINE_integer("insert_chunk_size", "1000",
                            'the number of rows to insert at a time.')
tf.app.flags.DEFINE_integer(
    "unknown_length", "50000",
    'the number of lines used to compute unknown_vector')
# tf.app.flags.DEFINE_integer(
#     "node_len", "100",
#     'the max #tokens of node string, 100 for 512 characters string.')
# tf.app.flags.DEFINE_integer(
#     "target_len", "200",
#     'the max #tokens of node string, 100 for 512 characters string.')
tf.app.flags.DEFINE_integer("html_len", "256",
                            'the number of tokens in one html string.')
tf.app.flags.DEFINE_integer("num_train_f", "4",
                            'number of training files per category.')
tf.app.flags.DEFINE_integer("num_test_f", "1",
                            'number of test files per category.')
tf.app.flags.DEFINE_integer("max_workers", "8", 'max num of threads')
tf.app.flags.DEFINE_string("dmoz_db", "dict", 'sqlite or dict')
tf.app.flags.DEFINE_string("wv_db", "dict", 'sqlite or dict')
tf.app.flags.DEFINE_boolean("verbose", False, "print extra thing")
tf.app.flags.DEFINE_integer("pages_per_file", "4000",
                            'number of web pages per TFRecords')

#########################################
# global variables
#########################################
CUR_TIME = time.strftime("%Y-%m-%d_%H-%M-%S")
# if FLAGS.dataset_type == "ukwa":
#     CATEGORIES = "Arts,Business,Computers,Health,Sports".split(',')
# elif FLAGS.dataset_type == "dmoz" and FLAGS.num_cats == 5:
#     CATEGORIES = "Arts,Business,Computers,Health,Sports".split(',')
# elif FLAGS.dataset_type == "dmoz" and FLAGS.num_cats == 10:
#     CATEGORIES = "Arts,Business,Computers,Health,Sports,Society,Science,Recreation,Shopping,Games".split(',')
# else:
#     raise ValueError("wrong num_cats: " + FLAGS.num_cats)
if FLAGS.dataset_type == 'dmoz':
    if FLAGS.num_cats == 5:
        CATEGORIES = ["Arts", "Business", "Computers", "Health", "Sports"]
    elif FLAGS.num_cats == 10:
        CATEGORIES = ["Arts", "Business", "Computers", "Health", 'Society',
                      'Science', 'Sports', 'Recreation', 'Shopping', 'Games']
        # CATEGORIES = list(set(["Arts", "Business", "Computers", "Health",
        #                        'Society', 'Science', 'Sports', 'Recreation',
        #                        'Shopping', 'Reference', 'Games', 'Home']))
    else:
        raise ValueError("cat_num wrong value: " + FLAGS.cat_num)
    DMOZ_JSON = "dmoz_{}.json".format(len(CATEGORIES))
elif FLAGS.dataset_type == 'ukwa':
    CATEGORIES = ['Arts & Humanities', 'Government, Law & Politics',
                  'Society & Culture', 'Business, Economy & Industry',
                  'Science & Technology', 'Medicine & Health',
                  'Education & Research', 'Company Web Sites',
                  'Digital Society', 'Sports and Recreation']
    DMOZ_JSON = "ukwa_{}.json".format(len(CATEGORIES))
else:
    raise ValueError("dataset_type has wrong value: {}".format(
        FLAGS.dataset_type))

# DMOZ_JSON = "dmoz_{}.json".format(len(CATEGORIES))
DMOZ_SQLITE = "dmoz.sqlite"

WV_FILE = 'glove.6B.50d.txt'
WV_SQLITE = 'wv.sqlite'

CAT_PREFIX = "Top/"
TFR_SUFFIX = '.TFR'

unknown_vector = []


def read_json(filename):
    if os.path.isfile(filename):
        logging.info("reading from json file:" + filename)
        with open(filename) as data_file:
            data = json.load(data_file)
        logging.info("finish reading json file")
        return data
    else:
        raise FileNotFoundError("json file:", filename)


def write_json(filename, data):
    logging.info("writing dmoz to" + filename)
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=4)
    logging.info("finish writing to" + filename)


def get_vocab():
    _ = maybe_download(FLAGS.datasets_dir, "vocab.txt", WV_DOWNLOAD, WV_URL)
    vocab_path = os.path.join(FLAGS.datasets_dir, "vocab.txt")
    vocab = list(open(vocab_path).readlines())
    vocab = [s.strip() for s in vocab]
    return vocab


def _int64_feature(v):
    value = [v]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """only for list of float, one dimension!"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _vec_str(vec, dtype=np.float32):
    """word vectors to numpy string."""
    logging.debug("wv to np: vec length: {}".format(len(vec)))
    # if len(vec):
    #     logging.debug("vec: {}".format(vec[0]))
    tmp = np.array(vec, dtype=dtype)
    return tmp.tostring()


def _vec_float(vec, dtype=np.float32):
    """word vectors to numpy string."""
    logging.debug("wv to np: vec length: {}".format(len(vec)))
    # if len(vec):
    #     logging.debug("vec: {}".format(vec[0]))
    tmp = np.array(vec, dtype=dtype)
    return np.reshape(tmp, (-1)).tolist()


def write_tfr(tfr_file, cat_id, target_wv, unlabeled_wv, labeled_p):
    """write pages to single TFRecords file.
    args:
        tfr_file (str): path of output file.
        cat_id (int): id of categories(label).
    return:
        none
    """
    num_examples = len(target_wv)

    logging.info('\nWriting' + tfr_file)
    writer = tf.python_io.TFRecordWriter(tfr_file)
    for index in range(num_examples):
        example = tf.train.Example(features=tf.train.Features(feature={
            # 'label': _int64_feature(cat_id),
            # 'target': _float_feature(target_wv[index]),
            # 'unlabeled': _float_feature(unlabeled_wv[index]),
            # 'labeled': _int64_feature(labeled_p[index])
            'label': _int64_feature(cat_id[index]),
            'target': _bytes_feature(_vec_str(target_wv[index])),
            'unlabeled': _float_feature(_vec_float(unlabeled_wv[index])),
            'un_len': _int64_feature(len(unlabeled_wv[index])),
            'labeled': _float_feature(_vec_float(labeled_p[index])),
            'la_len': _int64_feature(len(labeled_p[index]))
        }))
        writer.write(example.SerializeToString())
    logging.info("finish writing TFRecords file\n\n")


def convert_wv(target_p, unlabeled_p, wv_conn):
    """convert text strings into word vectors lists.
    args:
        target_p ([html_str])
        unlabeled_p ([[html_str]])
    return:
        target_wv ([[wv]])
        unlabeled_wv ([[[wv]]])
    """
    logging.info("\nconvert to word vector")
    target_wv = []
    for s in target_p:
        target_wv.append(wv_conn.select(s, str_len=FLAGS.html_len))

    unlabeled_wv = []
    for nodes in unlabeled_p:
        unlabeled_wv.append([])
        for s_node in nodes:
            # logging.debug("node length: {}".format(len(s_node)))
            unlabeled_wv[-1].append(wv_conn.select(s_node,
                                                   str_len=FLAGS.html_len))

    return target_wv, unlabeled_wv


def one_hot(cat):
    ret = [0] * len(CATEGORIES)
    ret[cat] = 1
    return ret


def _clean_url(url):
    """clean origianl url.
    only keep first level folder.
    like: www.yuhui.com/haha
    """
    try:
        url_p = urlparse(url)
        netloc = url_p.netloc.split(':')[0]
        # get the first level path
        path = ''
        if url_p.path:
            # print ("url:", url)
            path = url_p.path.split('/')[1]
        url_clean = netloc + '/' + path
    except ValueError:
        logging.info("exception: {}".format(url))
        return url
    return url_clean


def split_single_page(page, dmoz_conn):
    """split a single web page data."""
    label = page['label']
    target = page['html']
    unlabeled = []
    labeled = []

    logging.info("split page id: " + str(page['id']))
    logging.debug("#nodes: " + str(len(page['relatives'])))
    for node in page['relatives']:
        cat = dmoz_conn.select(_clean_url(node['r_url']))
        if cat >= 0:
            # labeled
            labeled.append(one_hot(cat))
        else:
            # unlabeled
            unlabeled.append(node['r_html'])
    return target, unlabeled, labeled, label


def split_pages(pages, dmoz_conn):
    """split pages into three lists by looking up Dmoz DB.
    args:
        pages ([page]): list of pages.
    return:
        target_p ([html_str])
        unlabeled_p ([[html_str]])
        labeled_p ([[labels in one hot encoding]])
    """
    logging.info("splitting nodes of pages")
    labels = []
    target_p = []
    unlabeled_p = []
    labeled_p = []

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=FLAGS.max_workers) as executor:
        # Start the load operations and mark each future with its URL
        future_page = {
            executor.submit(split_single_page, page, dmoz_conn): page
            for page in pages
        }
        for future in concurrent.futures.as_completed(future_page):
            p = future_page[future]
            try:
                result = future.result()
            except Exception as exc:
                logging.error('%r:%r generated an exception: %s' %
                              (p["id"], p["url"], exc))
            else:
                target_p.append(result[0])
                unlabeled_p.append(result[1])
                labeled_p.append(result[2])
                labels.append(result[3])
    return target_p, unlabeled_p, labeled_p, labels


class DmozDB(object):
    def __init__(self, db):
        if db == 'sqlite':
            self.init = self.init_sqlite
            self.select = self.select_sqlite
            self.close = self.close_sqlite
        elif db == 'dict':
            self.init = self.init_dict
            self.select = self.select_dict
            self.close = self.close_dict
        else:
            raise ValueError("wrong DB name: " + db)

        self.init()

    def init_sqlite(self):
        """get sqlite connection giving path of database file.
        args:
            sqlite_file (str): path of sqlite database file.
        return:
            connection object
        """
        sqlite_file = os.path.join(FLAGS.data_dir, DMOZ_SQLITE)
        logging.info("get dmoz sqlite connection")
        if os.path.isfile(sqlite_file):
            logging.info("DB already exists: " + sqlite_file)
            # conn = sqlite3.connect(sqlite_file, check_same_thread=False)

            # Read database to tempfile
            con = sqlite3.connect(sqlite_file)
            tempfile = StringIO()
            for line in con.iterdump():
                tempfile.write('%s\n' % line)
            con.close()
            tempfile.seek(0)

            # Create a database in memory and import from tempfile
            conn = sqlite3.connect(":memory:", check_same_thread=False)
            conn.cursor().executescript(tempfile.read())
            conn.commit()
            conn.row_factory = sqlite3.Row
        else:
            dmoz_json_path = os.path.join(FLAGS.data_dir, DMOZ_JSON)
            if not os.path.isfile(dmoz_json_path):
                raise FileNotFoundError(dmoz_json_path)
            dmoz = read_json(dmoz_json_path)

            logging.info("creating new database file: " + sqlite_file)
            # enable Serialized, only read is used.
            conn = sqlite3.connect(sqlite_file, check_same_thread=False)
            # Create table
            conn.execute('CREATE TABLE dmoz (url TEXT, category INTEGER)')
            # conn.execute('CREATE TABLE dmoz (url TEXT PRIMARY KEY, category INTEGER)')
            logging.info("created table")

            row_ind = 0
            url_chunk = []
            for cat in dmoz:
                for page in dmoz[cat]:
                    # url = urlparse(page['url'])
                    # # garentee / after netloc
                    # path = '/'
                    # if url.path:
                    #     path = url.path
                    url_clean = _clean_url(page['url'])
                    url_chunk.append((url_clean, CATEGORIES.index(cat)))
                    row_ind += 1
                    if row_ind % FLAGS.insert_chunk_size == 0:
                        # sql insert many rows
                        # logging.debug("url_chunk: {}".format(url_chunk))
                        conn.executemany('INSERT INTO dmoz VALUES (?,?)',
                                         url_chunk)
                        url_chunk = []
                        conn.commit()
                        logging.info(
                            "row {}: inset {} rows to dmoz TABLE".format(
                                row_ind, FLAGS.insert_chunk_size))
            # insert the last block
            if url_chunk:
                conn.executemany('INSERT INTO dmoz VALUES (?,?)', url_chunk)
                url_chunk = []
                conn.commit()
                logging.info("row {}: inset {} rows to dmoz TABLE".format(
                    row_ind, len(url_chunk)))

            # create index for url
            logging.info("creating url index for dmonz DB")
            conn.execute('CREATE INDEX url_index ON dmoz (url)')

        # def regexp(expr, item):
        #     reg = re.compile(expr)
        #     return reg.search(item) is not None
        #
        # conn.create_function("REGEXP", 2, regexp)

        self.conn = conn

    def init_dict(self):
        logging.info("creating dmoz dict")
        dmoz_json_path = os.path.join(FLAGS.data_dir, DMOZ_JSON)
        if not os.path.isfile(dmoz_json_path):
            raise FileNotFoundError(dmoz_json_path)
        dmoz = read_json(dmoz_json_path)
        self.dict = {}
        for cat in dmoz:
            cat_id = CATEGORIES.index(cat)
            for page in dmoz[cat]:
                url_clean = _clean_url(page['url'])
                self.dict[url_clean] = cat_id

    def select_sqlite(self, url):
        """check if url exists in Dmoz DB.
        args:
            url(str): string of url.
        return:
            index of categories id. -1 if not exists.
        """
        # LIKE is faster than REGEXP
        # url_regex = ['%' + netloc + '/' + path + '%']
        # exe = self.conn.execute("SELECT * FROM dmoz WHERE url LIKE ?", url_regex)
        # url_regex = [netloc + '/' + path]
        url_clean = _clean_url(url)
        exe = dmoz_conn.execute("SELECT * FROM dmoz WHERE url = ?", url_clean)
        # don't use regex, way too slow
        # url_regex = ['.*' + netloc + '/' + path + '.*']
        # exe = dmoz_conn.execute("SELECT * FROM dmoz WHERE url REGEXP ?", url_regex)
        result = exe.fetchone()
        if result:
            logging.debug("find regex: " + str(result))
            # if found, return category id
            return result[1]
        return -1

    def select_dict(self, url):
        if url in self.dict:
            logging.debug("select dmoz: " + url)
            return self.dict[url]
        else:
            return -1

    def close_sqlite(self):
        self.conn.close()

    def close_dict(self):
        pass


class WvDB(object):
    def __init__(self, db):
        if db == 'sqlite':
            self.init = self.init_sqlite
            self.select = self.select_sqlite
            self.close = self.close_sqlite
        elif db == 'dict':
            self.init = self.init_dict
            self.select = self.select_dict
            self.close = self.close_dict
        else:
            raise ValueError("wrong DB name: " + db)

        self.init()

    def init_sqlite(self):
        """get sqlite connection giving path of database file.
        args:
            sqlite_file (str): path of sqlite database file.
        return:
            connection object
        """
        sqlite_file = os.path.join(FLAGS.data_dir, WV_SQLITE)
        logging.info("get wv sqlite connection")
        global unknown_vector
        if os.path.isfile(sqlite_file):
            logging.info("DB already exists: " + sqlite_file)
            conn = sqlite3.connect(sqlite_file)
            exe = conn.execute("SELECT * FROM unknown")
            result = exe.fetchone()
            if result:
                logging.debug("find unknown_vector: " + result[0])
                unknown_vector = list(map(float, result[0].split(' ')))
            else:
                raise ValueError("cannot find unknown_vector in DB")
        else:
            wv_file_path = os.path.join(FLAGS.data_dir, WV_FILE)
            if not os.path.isfile(wv_file_path):
                raise FileNotFoundError("word vector file: " + wv_file_path)

            logging.info("creating new database file: " + sqlite_file)
            conn = sqlite3.connect(sqlite_file)
            # Create table, primary key make insert slow! but may accelerate select
            conn.execute(
                'CREATE TABLE wv (word TEXT PRIMARY KEY, vector TEXT, id INT)')
            conn.execute('CREATE TABLE unknown (vector TEXT)')
            logging.info("created table")

            word_chunk = []
            unknown = [0] * FLAGS.we_dim
            with open(wv_file_path) as lines:
                for row_ind, line in enumerate(lines):
                    tmp = line.split(' ', 1)
                    word = tmp[0]
                    vector = tmp[1]
                    word_chunk.append((word, vector, row_ind))

                    # sum all word vectors
                    if row_ind < FLAGS.unknown_length:
                        unknown = [
                            x + y
                            for x, y in zip(unknown, list(map(
                                float, vector.split(' '))))
                        ]

                    if row_ind % FLAGS.insert_chunk_size == 0:
                        # sql insert many rows
                        conn.executemany('INSERT INTO wv VALUES (?,?,?)',
                                         word_chunk)
                        word_chunk = []
                        conn.commit()
                        logging.info(
                            "row {}: inset {} rows to wv TABLE".format(
                                row_ind, FLAGS.insert_chunk_size))
                        # break

                        # insert the last block of word vector
                if word_chunk:
                    conn.executemany('INSERT INTO wv VALUES (?,?,?)',
                                     word_chunk)
                    word_chunk = []
                    conn.commit()
                    logging.info("row {}: inset {} rows to wv TABLE".format(
                        row_ind, len(word_chunk)))

                # get average of all word vectors
                unknown_vector = [x / FLAGS.unknown_length for x in unknown]
                logging.info("unknown_vector: {}".format(unknown_vector))
                tmp = tuple([' '.join(list(map(str, unknown_vector)))])
                # logging.debug(tmp)
                conn.execute('INSERT INTO unknown VALUES (?)', tmp)
                conn.commit()
        self.conn = conn

    def init_dict(self):
        logging.info("get wv dict connection")
        wv_file_path = os.path.join(FLAGS.data_dir, WV_FILE)
        if not os.path.isfile(wv_file_path):
            raise FileNotFoundError("word vector file: " + wv_file_path)

        logging.info("creating wv dict")
        self.dict = {}
        unknown = [0] * FLAGS.we_dim
        with open(wv_file_path) as lines:
            for row_ind, line in enumerate(lines):
                tmp = line.split(' ', 1)
                word = tmp[0]
                vector = tmp[1]
                v_float = list(map(float, vector.split(" ")))
                self.dict[word] = v_float

                # sum all word vectors
                if row_ind < FLAGS.unknown_length:
                    unknown = [x + y for x, y in zip(unknown, v_float)]

        # get average of all word vectors
        self.unknown = [x / FLAGS.unknown_length for x in unknown]
        logging.info("unknown_vector: {}".format(self.unknown))

    def select_sqlite(self, string, str_len):
        """convert a string to word vectors.
        args:
            string (str): string.
            wv_conn: sqlite connection to wv DB.
            str_len (int): number of word vectors in one string.
        return:
            [word_vectors]
        """
        vectors = []
        # tokenize html string
        s_tokens = nltk.word_tokenize(string.lower())
        logging.debug("s_tokens length: {}".format(len(s_tokens)))
        for token in s_tokens:
            exe = self.conn.execute("SELECT * FROM wv WHERE word = ?",
                                    tuple([token]))
            result = exe.fetchone()
            if result:
                # logging.debug("find regex: " + str(result))
                vector = list(map(float, result[1].split(" ")))
            else:
                # unknown word
                logging.debug("unknown word: {}".format(token))
                vector = unknown_vector
            vectors.append(vector)
            # logging.debug("vector: {}".format(vector))
            # align node length

        if str_len >= 0:
            if len(vectors) > str_len:
                vectors = vectors[:str_len]
            else:
                vectors.extend([unknown_vector] * (str_len - len(vectors)))
        return vectors

    def select_dict(self, string, str_len):
        """convert a string to word vectors.
        args:
            string (str): string.
            wv_conn: sqlite connection to wv DB.
            str_len (int): number of word vectors in one string.
        return:
            [word_vectors]
        """
        vectors = []
        num_known = 0
        # tokenize html string
        s_tokens = nltk.word_tokenize(string.lower())
        logging.debug("s_tokens length: {}".format(len(s_tokens)))
        # slice long string
        if str_len >= 0:
            if len(s_tokens) > str_len:
                s_tokens = s_tokens[:str_len]
        # print("tokens: {}".format(s_tokens))
        tokens_test = []
        for token in s_tokens:
            # vector = self.dict.get(token, self.unknown)
            if token in self.dict:
                vector = self.dict[token]
                num_known += 1
                tokens_test.append(token)
            else:
                vector = self.unknown
            # vector = self.unknown
            vectors.append(vector)
            # logging.debug("vector: {}".format(vector))

            # pad node length
        if str_len >= 0:
            if len(vectors) < str_len:
                vectors.extend([self.unknown] * (str_len - len(vectors)))
            # print("tokens_test: {}".format(tokens_test))
            # print("s_tokens len: {}".format(len(s_tokens)))
            if FLAGS.verbose:
                logging.info("num_known/total_words = {}".format(
                    num_known / len(s_tokens)))
        return vectors

    def close_sqlite(self):
        self.conn.close()

    def close_dict(self):
        pass


def set_logging(stream=False, fileh=False, filename="example.log"):
    """set basic logging configurations (root logger).
    args:
        stream (bool): whether logging.info log to console.
        fileh (bool): whether write log to file.
        filename (str): the path of log file.
    return:
        configued root logger.
    """
    handlers = []
    level = logging.INFO
    log_format = '%(asctime)s: %(message)s'

    if stream:
        handlers.append(logging.StreamHandler())
    if fileh:
        handlers.append(logging.FileHandler(filename))
    logging.basicConfig(format=log_format, handlers=handlers, level=level)
    return logging.getLogger()


def main(argv):
    print("start of main\n")

    # file handling
    if not os.path.isdir(FLAGS.data_dir):
        raise FileNotFoundError("data_dir doesn't exist: " + FLAGS.data_dir)

    html_dir = os.path.join(FLAGS.data_dir, FLAGS.html_folder)
    if not os.path.isdir(html_dir):
        raise FileNotFoundError("html_folder doesn't exist: " +
                                FLAGS.html_folder)

    tfr_dir = os.path.join(FLAGS.data_dir, "TFR_" + CUR_TIME)
    os.mkdir(tfr_dir)
    train_dir = os.path.join(tfr_dir, 'train')
    os.mkdir(train_dir)
    test_dir = os.path.join(tfr_dir, 'test')
    os.mkdir(test_dir)
    shuf_dir = os.path.join(tfr_dir, 'shuffle')
    os.mkdir(shuf_dir)

    # loging
    log_file = os.path.join(tfr_dir, "log")
    set_logging(stream=True, fileh=True, filename=log_file)
    logging.info("\nall arguments:")
    for attr, value in sorted(FLAGS.__flags.items()):
        logging.info("{}={}".format(attr.upper(), value))
    logging.info("")

    # shuffle all data thoroughly
    # all data stored in several train and test json files.
    train_set = []
    test_set = []
    logging.info('')
    logging.info('reading all data, split into train and test')
    for category in os.listdir(html_dir):
        cat_dir = os.path.join(html_dir, category)
        if os.path.isdir(cat_dir):
            cat_id = CATEGORIES.index(category)
            ind = 0
            for j_file in os.listdir(cat_dir):
                if ind < FLAGS.num_train_f + FLAGS.num_test_f and not j_file.startswith(
                        '.'):
                    # read single html json file
                    j_path = os.path.join(cat_dir, j_file)
                    pages = read_json(j_path)
                    for page in pages:
                        page['label'] = cat_id

                    # write to a single TFRecords file
                    if ind < FLAGS.num_train_f:
                        train_set.extend(pages)
                    else:
                        test_set.extend(pages)
                    ind += 1

    # shuffle all data
    logging.info("\nshuffling train set")
    shuffle(train_set)
    logging.info("\nshuffling test set")
    shuffle(test_set)

    logging.info("\nwriting shuffled train data into json")
    for i in range(math.ceil(len(train_set) / FLAGS.pages_per_file)):
        head = i * FLAGS.pages_per_file
        tail = (i + 1) * FLAGS.pages_per_file
        if tail > len(train_set):
            tail = -1
        p = os.path.join(shuf_dir, str(i) + ".train")
        write_json(p, train_set[head:tail])

    logging.info("\nwriting shuffled test data into json")
    for i in range(math.ceil(len(test_set) / FLAGS.pages_per_file)):
        head = i * FLAGS.pages_per_file
        tail = (i + 1) * FLAGS.pages_per_file
        if tail > len(test_set):
            tail = -1
        p = os.path.join(shuf_dir, str(i) + ".test")
        write_json(p, test_set[head:tail])

    # local variables
    dmoz_conn = DmozDB(FLAGS.dmoz_db)
    wv_conn = WvDB(FLAGS.wv_db)
    train_index = 0
    test_index = 0

    # reading json and convert to TFRecords
    logging.info('\n\nconverting to TFRecords')
    for s_file in os.listdir(shuf_dir):
        j_path = os.path.join(shuf_dir, s_file)
        if os.path.isfile(j_path) and not s_file.startswith('.'):
            logging.info('')
            logging.info('train:{}, test:{}'.format(train_index, test_index))
            # read single html json file
            pages = read_json(j_path)

            # check labeled relative nodes by looking up dmoz DB
            target_p, unlabeled_p, labeled_p, labels = split_pages(pages,
                                                                   dmoz_conn)

            # map word to word vector by looking up WE DB
            target_wv, unlabeled_wv = convert_wv(target_p, unlabeled_p,
                                                 wv_conn)

            # write to a single TFRecords file
            if s_file.endswith(".train"):
                tfr_file = os.path.join(train_dir,
                                        str(train_index) + TFR_SUFFIX)
                train_index += 1
            elif s_file.endswith(".test"):
                tfr_file = os.path.join(test_dir, str(test_index) + TFR_SUFFIX)
                test_index += 1
            else:
                raise ValueError("wrong s_file")

            write_tfr(tfr_file, labels, target_wv, unlabeled_wv, labeled_p)

    dmoz_conn.close()
    wv_conn.close()

    print("\n end of main~~~")


if __name__ == '__main__':
    tf.app.run()
