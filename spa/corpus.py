import os
import re


class Corpus:
    def __init__(self, filepath):
        root_path = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.normpath(os.path.join(root_path, filepath))

        re_split = re.compile("\s+")

        self.pos_doc_list = []
        self.neg_doc_list = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                splits = re_split.split(line.strip())
                if splits[0] == "pos":
                    self.pos_doc_list.append(splits[1:])
                elif splits[0] == "neg":
                    self.neg_doc_list.append(splits[1:])
                else:
                    raise ValueError("Corpus Error")

        self.doc_length = len(self.pos_doc_list)
        assert len(self.neg_doc_list) == self.doc_length

        self.train_num = 0
        self.test_num = 0

        runout_content = "You are using the corpus: %s.\n" % filepath
        runout_content += "There are total %d positive and %d negative f_corpus." % \
                          (self.doc_length, self.doc_length)
        print(runout_content)

    def get_corpus(self, start=0, end=-1):
        assert self.doc_length >= self.test_num + self.train_num

        if end == -1:
            end = self.doc_length

        data = self.pos_doc_list[start:end] + self.neg_doc_list[start:end]
        data_labels = [1] * (end - start) + [0] * (end - start)
        return data, data_labels

    def get_train_corpus(self, num):
        self.train_num = num
        return self.get_corpus(end=num)

    def get_test_corpus(self, num):
        self.test_num = num
        return self.get_corpus(start=self.train_num, end=self.train_num + num)

    def get_all_corpus(self):
        data = self.pos_doc_list[:] + self.neg_doc_list[:]
        data_labels = [1] * self.doc_length + [0] * self.doc_length
        return data, data_labels








class WaimaiCorpus(Corpus):
    def __init__(self):
        Corpus.__init__(self, "E:\Desktop\SentimentPolarityAnalysis-master\spa/f_corpus/ch_waimai_corpus.txt")











def get_waimai_corpus():
    from oujago.seg import cut

    origin_filepath = ['D:\\My Data\\NLP\\SA\\waimai\\negative_corpus.txt',
                       'D:\\My Data\\NLP\\SA\\waimai\\positive_corpus.txt']
    corpus_filepath = "f_corpus/ch_waimai2_corpus.txt"

    origin_filepath = ['f_corpus/waimai/negative_corpus_v1.txt',
                       'f_corpus/waimai/positive_corpus_v1.txt']
    corpus_filepath = "f_corpus/ch_waimai_corpus.txt"

    labels = ["neg", 'pos']

    with open(corpus_filepath, "w", encoding="utf-8") as write_f:
        for i in range(len(labels)):
            j = 0
            with open(origin_filepath[i], encoding="utf-8") as read_f:
                for line in read_f:
                    write_f.write("%s\t%s\n" % (labels[i], "\t".join(cut(line.strip()))))
                    j += 1
                    if j == 4000:
                        break


def test_corpus():
    a = WaimaiCorpus()
    pass


