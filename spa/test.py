import datetime
from multiprocessing import Process

from spa.feature_extraction import ChiSquare
from spa.tools import get_accuracy
from spa.tools import Write2File


class Test:
    def __init__(self, type_, train_num, test_num, feature_num, max_iter, C, k, corpus):
        self.type = type_
        self.train_num = train_num
        self.test_num = test_num
        self.feature_num = feature_num
        self.max_iter = max_iter
        self.C = C
        self.k = k
        self.parameters = [train_num, test_num, feature_num]

        # get the f_corpus
        self.train_data, self.train_labels = corpus.get_train_corpus(train_num)
        self.test_data, self.test_labels = corpus.get_test_corpus(test_num)

        # feature extraction
        fe = ChiSquare(self.train_data, self.train_labels)
        self.best_words = fe.best_words(feature_num)

        self.single_classifiers_got = False

        self.precisions = [[0, 0],  # bayes
                           [0, 0],  # maxent
                           [0, 0]]  # svm

    def set_precisions(self, precisions):
        self.precisions = precisions



    def test_bayes(self):
        print("BayesClassifier")
        print("---" * 45)
        print("Train num = %s" % self.train_num)
        print("Test num = %s" % self.test_num)

        from spa.classifiers import BayesClassifier
        bayes = BayesClassifier(self.train_data, self.train_labels, self.best_words)

        classify_labels = []
        print("BayesClassifier is testing ...")
        for data in self.test_data:
            classify_labels.append(bayes.classify(data))
        print("BayesClassifier tests over.")

        filepath = "E:\Desktop\SentimentPolarityAnalysis-master\Bayes.xls"

        self.write(filepath, classify_labels, 0)

    def write(self, filepath, classify_labels, i=-1):
        results = get_accuracy(self.test_labels, classify_labels, self.parameters)
        if i >= 0:
            self.precisions[i][0] = results[10][1] / 100
            self.precisions[i][1] = results[7][1] / 100

        Write2File.write_contents(filepath, results)



    def test_svm(self):
        print("SVMClassifier")
        print("---" * 45)
        print("Train num = %s" % self.train_num)
        print("Test num = %s" % self.test_num)
        print("C = %s" % self.C)

        from spa.classifiers import SVMClassifier
        svm = SVMClassifier(self.train_data, self.train_labels, self.best_words, self.C)

        classify_labels = []
        print("SVMClassifier is testing ...")
        for data in self.test_data:
            classify_labels.append(svm.classify(data))
        print("SVMClassifier tests over.")

        filepath = "E:\Desktop\SentimentPolarityAnalysis-master\SVM.xls"

        self.write(filepath, classify_labels, 2)




def test_waimai():
    from spa.corpus import WaimaiCorpus

    type_ = "waimai"
    train_num = 3000
    test_num = 1000
    feature_num = 5000
    max_iter = 500
    C = 150
    k = 13
    k = [1, 3, 5, 7, 9, 11, 13]
    corpus = WaimaiCorpus()

    test = Test(type_, train_num, test_num, feature_num, max_iter, C, k, corpus)

    # test.single_multiprocess()
    # test.multiple_multiprocess()

    # test.test_knn()
    test.test_bayes()
    # test.test_maxent()
    # test.test_maxent_iteration()
    test.test_svm()
    # test.test_multiple_classifiers()
    # test.test_multiple_classifiers2()
    # test.test_multiple_classifiers3()
    # test.test_multiple_classifiers4()





