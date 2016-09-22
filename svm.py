from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
# import sys
from time import time
import matplotlib.pyplot as plt
import os
import argparse as ap

# from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import density
from sklearn import metrics


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
opts = {}
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
opts['print_report'] = True
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
opts['select_chi2'] = 3
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
opts['print_cm'] = True
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
opts['print_top10'] = True
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
opts['all_categories'] = True
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
opts['use_hashing'] = True
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
opts['n_features'] = 2 ** 16
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")
opts['filtered'] = False

op.add_option("--dataset",
              action="store", type=str, default="dmoz-5",
              help="n_features when using the hashing vectorizer.")
opts['dataset'] = 'dmoz-5'
opts = ap.Namespace(**opts)


# (opts, args) = op.parse_args()
# if len(args) > 0:
#     op.error("this script takes no arguments.")
#     sys.exit(1)
#
# print(__doc__)
# op.print_help()
# print()


###############################################################################
# Load some categories from the training set
root_path = "/Users/yuhui.lin/work/fastText/data/"
if opts.dataset == "dmoz-5":
    data_path = os.path.join(root_path, "TFR_5-fast")
    num_cats = 5
elif opts.dataset == "dmoz-10":
    data_path = os.path.join(root_path, "TFR_10-fast")
    num_cats = 10
elif opts.dataset == "ukwa":
    data_path = os.path.join(root_path, "TFR_ukwa-fast")
    num_cats = 10
else:
    raise ValueError(opts.dataset)

if opts.all_categories:
    categories = None
else:
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]

if opts.filtered:
    remove = ('headers', 'footers', 'quotes')
else:
    remove = ()

print("Loading 20 newsgroups dataset for categories:")
print(categories if categories else "all")

# data_train = fetch_20newsgroups(subset='train', categories=categories,
#                                 shuffle=True, random_state=42,
#                                 remove=remove)
#
# data_test = fetch_20newsgroups(subset='test', categories=categories,
#                                shuffle=True, random_state=42,
#                                remove=remove)

def svm(num_cats, train_data, test_data):
    # train_path = os.path.join(data_path, "train")
    # test_path = os.path.join(data_path, "test")

    data_train = {}
    data_train["data"] = []
    data_train["target"] = []
    data_train["target_names"] = [str(i) for i in range(num_cats)]
    data_test = {}
    data_test["data"] = []
    data_test["target"] = []
    # with open(train_path, 'r') as train_f, open(test_path) as test_f:
    #     train_data = train_f.readlines()
    #     test_data = test_f.readlines()

    for exam in train_data:
        data_train["data"].append(exam[12:])
        # print(exam)
        data_train["target"].append(int(exam[9]))
    for exam in test_data:
        data_test["data"].append(exam[12:])
        data_test["target"].append(int(exam[9]))


    data_train = ap.Namespace(**data_train)
    data_test = ap.Namespace(**data_test)

    print('data loaded')

    categories = data_train.target_names    # for case categories == None


    def size_mb(docs):
        return sum(len(s.encode('utf-8')) for s in docs) / 1e6

    data_train_size_mb = size_mb(data_train.data)
    data_test_size_mb = size_mb(data_test.data)

    print("%d documents - %0.3fMB (training set)" % (
        len(data_train.data), data_train_size_mb))
    print("%d documents - %0.3fMB (test set)" % (
        len(data_test.data), data_test_size_mb))
    print("%d categories" % len(categories))
    print(data_train.data[:10])

    # split a training set and a test set
    y_train, y_test = data_train.target, data_test.target

    print("Extracting features from the training data using a sparse vectorizer")
    t0 = time()
    if opts.use_hashing:
        vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                    n_features=opts.n_features)
        X_train = vectorizer.transform(data_train.data)
    else:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                    stop_words='english')
        X_train = vectorizer.fit_transform(data_train.data)
    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()

    print("Extracting features from the test data using the same vectorizer")
    t0 = time()
    X_test = vectorizer.transform(data_test.data)
    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print()

    # mapping from integer feature name to original token string
    if opts.use_hashing:
        feature_names = None
    else:
        feature_names = vectorizer.get_feature_names()

    if opts.select_chi2:
        print("Extracting %d best features by a chi-squared test" %
            opts.select_chi2)
        t0 = time()
        ch2 = SelectKBest(chi2, k=opts.select_chi2)
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)
        if feature_names:
            # keep selected feature names
            feature_names = [feature_names[i] for i
                            in ch2.get_support(indices=True)]
        print("done in %fs" % (time() - t0))
        print()

    if feature_names:
        feature_names = np.asarray(feature_names)


    def trim(s):
        """Trim string to fit on terminal (assuming 80-column display)"""
        return s if len(s) <= 80 else s[:77] + "..."


    ###############################################################################
    # Benchmark classifiers
    def benchmark(clf):
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(X_train, y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        t0 = time()
        pred = clf.predict(X_test)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        score = metrics.accuracy_score(y_test, pred)
        print("accuracy:   %0.3f" % score)

        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])
            print("density: %f" % density(clf.coef_))

            if opts.print_top10 and feature_names is not None:
                print("top 10 keywords per class:")
                for i, category in enumerate(categories):
                    top10 = np.argsort(clf.coef_[i])[-10:]
                    print(trim("%s: %s"
                        % (category, " ".join(feature_names[top10]))))
            print()

        if opts.print_report:
            print("classification report:")
            print(metrics.classification_report(y_test, pred,
                                                target_names=categories))

        if opts.print_cm:
            print("confusion matrix:")
            print(metrics.confusion_matrix(y_test, pred))

        print()
        clf_descr = str(clf).split('(')[0]
        return clf_descr, score, train_time, test_time


    results = []
    for penalty in ["l2", "l1"]:
        print('=' * 80)
        print("%s penalty" % penalty.upper())
        # Train Liblinear model
        results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                                dual=False, tol=1e-3)))



    # make some plots

    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(4)]

    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, .2, label="score", color='r')
    plt.barh(indices + .3, training_time, .2, label="training time", color='g')
    plt.barh(indices + .6, test_time, .2, label="test time", color='b')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)

    plt.show()
