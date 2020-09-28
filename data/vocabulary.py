import json
from .tokenizer import Tokenizer, _Tokenizer

class VocabDict(object):
    def __init__(self, data_path, out_path, max_n_words = 30000):
        self.data_path = data_path
        self.out_path = out_path

        self.max_n_words = max_n_words
    
    def _file_handle(self, file):
        pass

    def generate_vocabfile(self, data_path = None, out_path = None):
        dic = dict()
        idx = 0

        if data_path:
            self.data_path = data_path
        if out_path:
            self.out_path = out_path

        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                words = line.strip().split()
                for w in words:
                    if w in dic:
                        dic[w] += 1
                    else:
                        dic[w] = 1
        
        items = sorted(dic.items(), key = lambda d: d[1], reverse=True)[:self.max_n_words]

        dic.clear()        
        for word, num in items:
            dic[word] = (idx, num)
            idx += 1

        with open(self.out_path + '.json', 'w', encoding='utf-8') as f:
            json.dump(dic, f, indent=4, ensure_ascii=False)

    def to_json(self, vocab_path, out_path):
        dic = {}

        print('begin reading the file')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                words = line.strip().split()
                dic[words[0]] = (i, words[1])
        
        print('read successful!')
        print('begin trans')
        
        with open(out_path + '.json', 'w', encoding='utf-8') as f:
            json.dump(dic, f, indent=4, ensure_ascii=False)

        print('trans successful')    


class Vocabulary(object):
    PAD = 0
    EOS = 1
    BOS = 2
    UNK = 3

    def __init__(self, type, dict_path, max_n_words=-1):

        self.dict_path = dict_path
        self._max_n_words = max_n_words

        self._load_vocab(self.dict_path)
        self._id2token = dict([(ii[0], ww) for ww, ii in self._token2id_feq.items()])
        self.tokenizer = Tokenizer(type=type)  # type: _Tokenizer

    @property
    def max_n_words(self):

        if self._max_n_words == -1:
            return len(self._token2id_feq)
        else:
            return self._max_n_words

    def _init_dict(self):

        return {
            "<PAD>": (self.PAD, 0),
            "<UNK>": (self.UNK, 0),
            "<EOS>": (self.EOS, 0),
            "<BOS>": (self.BOS, 0)
        }

    def _load_vocab(self, path):
        """
        Load vocabulary from file

        If file is formatted as json, for each item the key is the token, while the value is a tuple such as
        (word_id, word_feq), or a integer which is the index of the token. The index should start from 0.

        If file is formatted as a text file, each line is a token
        """
        self._token2id_feq = self._init_dict()
        N = len(self._token2id_feq)

        if path.endswith(".json"):

            with open(path, encoding='utf-8') as f:
                _dict = json.load(f)
                # Word to word index and word frequence.
                for ww, vv in _dict.items():
                    if isinstance(vv, int):
                        self._token2id_feq[ww] = (vv + N, 0)
                    else:
                        self._token2id_feq[ww] = (vv[0] + N, vv[1])
        else:
            with open(path) as f:
                for i, line in enumerate(f):
                    ww = line.strip().split()[0]
                    self._token2id_feq[ww] = (i + N, 0)

    def word2id(self, word):

        if word in self._token2id_feq and self._token2id_feq[word][0] < self.max_n_words:
            return self._token2id_feq[word][0]
        else:
            return self.UNK

    def id2word(self, word_id):

        return self._id2token[word_id]

    def save2file(self, outfile):
        with open(outfile, 'w', encoding='utf-8') as file:
            for id, word in enumerate(self._id2token):
                if id > self.UNK: file.write(word + '\n')

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.BOS

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.PAD

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.EOS

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.UNK


PAD = Vocabulary.PAD
EOS = Vocabulary.EOS
BOS = Vocabulary.BOS
UNK = Vocabulary.UNK
