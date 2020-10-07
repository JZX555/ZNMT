import torch.nn.functional as F
from data.DataLoader import *
from module.Utils import *
from module.Embeddings import Embeddings
from module import Init
from data.Vocab import NMTVocab
from data.vocabulary import Vocabulary


class NMTHelper(object):
    def __init__(self, model, critic, src_vocab, tgt_vocab, config):
        self.model = model
        self.critic = critic
        
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        self.src_pad = src_vocab.pad()
        self.src_bos = src_vocab.bos()
        self.tgt_pad = tgt_vocab.pad()
        self.tgt_eos = tgt_vocab.eos()
        self.tgt_bos = tgt_vocab.bos()

        self.config = config
        self.src_fusion_list = None
        self.tgt_fusion_list = None

        self.src_extention_size = 0
        self.tgt_extention_size = 0

        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None

        # load extention pretrain embeddings
        if self.config.load_extention_vocab is True:
            self.load_extention_vocab()


    def transfer_embeddings(self, src_vocab:Vocabulary, tgt_vocab:Vocabulary, embedding:Embeddings):
        new_weight = embedding.embeddings.weight.new_zeros((tgt_vocab.max_n_words, embedding.embeddings.embedding_dim))
        cnt = 0

        with torch.no_grad():
            new_weight[src_vocab.PAD].fill_(0.0)

            for word, item in tgt_vocab._token2id_feq.items():
                if word in src_vocab._token2id_feq:
                    new_weight[item[0]] = embedding.embeddings.weight[src_vocab._token2id_feq[word][0]]
                    cnt += 1

            new_weight = torch.nn.Parameter(new_weight)
            embedding.embeddings.weight = new_weight

            print('restore {} word embeddings'.format(cnt))

    def load_extention_vocab(self):
        vocabs = self.config.extention_vocabs_path
        
        if vocabs is None:
            raise ValueError('Extention vocabs is None, can not load with none type')
        
        self.src_fusion_list = self.config.src_fusion_list
        self.tgt_fusion_list = self.config.tgt_fusion_list
        self.extention_embeddings_size = self.config.extention_embeddings_size
        self.extention_vocabs = {}
        self.extention_embeddings = {}

        for name, path in vocabs.items():
            vocab = Vocabulary(self.config.src_vocab_type, path)
            self.extention_vocabs[name] = vocab
            embedding = Embeddings(num_embeddings=vocab.max_n_words - 4, 
                                                    embedding_dim=self.extention_embeddings_size[name], 
                                                    dropout=self.config.dropout_emb,
                                                    add_position_embedding=self.config.add_position_emb)
            
            if name in self.config.extention_embeddings_path:
                state_dict = torch.load(self.config.extention_embeddings_path[name])
                # if name not in state_dict:
                #     print("Warning: {0} has no content saved!".format(name))
                # else:
                print("Loading {0}".format(name))
                embedding.embeddings.load_state_dict(state_dict)
            else:
                print("Warning: {0} has no content saved!".format(name))
            
            if name in self.src_fusion_list:
                print('begin load {}'.format(name))
                self.transfer_embeddings(vocab, self.src_vocab, embedding)
                self.src_extention_size += self.extention_embeddings_size[name]

            elif name in self.tgt_fusion_list:
                print('begin load {}'.format(name))
                self.transfer_embeddings(vocab, self.tgt_vocab, embedding)
            
            if name in self.tgt_fusion_list:
                self.tgt_extention_size += self.extention_embeddings_size[name]

            self.extention_embeddings[name] = embedding.cuda(self.device)
        
        
        # set embedding fuction for encoder and decoder
        self.model.set_embeddings_fuc(self.fusion_src_embeddings, self.fusion_tgt_embeddings, self.src_extention_size, self.tgt_extention_size)


    def prepare_training_data(self, src_inputs, tgt_inputs):
        self.train_data = []
        #for idx in range(self.config.max_train_length):
        self.train_data.append([])
        for src_input, tgt_input in zip(src_inputs, tgt_inputs):
            #idx = int(len(src_input) - 1)
            self.train_data[0].append((self.src_data_id(src_input), self.tgt_data_id(tgt_input)))
        self.train_size = len(src_inputs)
        self.batch_size = self.config.train_batch_size
        batch_num = 0
        #for idx in range(self.config.max_train_length):
        train_size = len(self.train_data[0])
        batch_num += int(np.ceil(train_size / float(self.batch_size)))
        self.batch_num = batch_num

    def prepare_valid_data(self, src_inputs, tgt_inputs):
        self.valid_data = []
        for src_input, tgt_input in zip(src_inputs, tgt_inputs):
            self.valid_data.append((self.src_data_id(src_input), self.tgt_data_id(tgt_input)))
        self.valid_size = len(self.valid_data)

    def src_data_id(self, src_input):
        result = [self.src_vocab.word2id(cur_word) for cur_word in src_input]

        return [self.src_bos] + result

    def tgt_data_id(self, tgt_input):
        result = [self.tgt_vocab.word2id(cur_word) for cur_word in tgt_input]

        return [self.tgt_bos] + result + [self.tgt_eos]

    def prepare_eval_data(self, src_inputs):
        eval_data = []
        for src_input in src_inputs:
            eval_data.append((self.src_data_id(src_input), src_input))

        return eval_data

    def pair_data_variable(self, batch):
        batch_size = len(batch)

        src_lengths = [len(batch[i][0]) for i in range(batch_size)]
        max_src_length = int(np.max(src_lengths))

        tgt_lengths = [len(batch[i][1]) for i in range(batch_size)]
        max_tgt_length = int(np.max(tgt_lengths))

        src_words = torch.zeros([batch_size, max_src_length], dtype=torch.int64, requires_grad=False)
        tgt_words = torch.zeros([batch_size, max_tgt_length], dtype=torch.int64, requires_grad=False)

        src_words = src_words.fill_(self.src_pad)
        tgt_words = tgt_words.fill_(self.tgt_pad)

        for b, instance in enumerate(batch):
            for index, word in enumerate(instance[0]):
                src_words[b, index] = word
            for index, word in enumerate(instance[1]):
                tgt_words[b, index] = word

        if self.use_cuda:
            src_words = src_words.cuda(self.device)
            tgt_words = tgt_words.cuda(self.device)

        return src_words, tgt_words, src_lengths, tgt_lengths

    def source_data_variable(self, batch):
        batch_size = len(batch)

        src_lengths = [len(batch[i][0]) for i in range(batch_size)]
        max_src_length = int(src_lengths[0])

        src_words = torch.zeros([batch_size, max_src_length], dtype=torch.int64, requires_grad=False)
        src_words = src_words.fill_(self.src_pad)

        for b, instance in enumerate(batch):
            for index, word in enumerate(instance[0]):
                src_words[b, index] = word


        if self.use_cuda:
            src_words = src_words.cuda(self.device)
        return src_words, src_lengths

    def fusion_src_embeddings(self, emb, seqs_x, fusion_list:list = None):

        if fusion_list is None:
            fusion_list = self.src_fusion_list

        if fusion_list is not None:
            for name in fusion_list:
                if name in self.extention_embeddings:
                    '''
                    please write the fusion method here.
                    which items is the list which contain the extention pretrain model name:
                    eg. ['BERT_src', 'Elmo_src', ...]
                    '''
                    emb = torch.cat([emb, self.extention_embeddings[name](seqs_x).detach()], dim=-1)

        return emb

    def fusion_tgt_embeddings(self, emb, fusion_list:list = None):

        if fusion_list is None:
            fusion_list = self.tgt_fusion_list

        if fusion_list is not None:
            for name in fusion_list:
                if name in self.extention_embeddings:
                    '''
                    please write the fusion method here.
                    which items is the list which contain the extention pretrain model name:
                    eg. ['BERT_tgt', 'Elmo_tgt', ...]
                    '''
                    emb = torch.cat([emb, self.extention_embeddings[name](seqs_x).detach()], dim=-1)

        return emb

    def compute_forward(self, seqs_x, seqs_y, xlengths, normalization=1.0):
        """
        :type model: Transformer

        :type critic: NMTCritierion
        """


        y_inp = seqs_y[:, :-1].contiguous()
        y_label = seqs_y[:, 1:].contiguous()

        with torch.enable_grad():
            logits = self.model(seqs_x, y_inp, lengths=xlengths)

            loss = self.critic(inputs=logits,
                            labels=y_label,
                            normalization=normalization)

            loss = loss.sum()
        torch.autograd.backward(loss)

        mask = y_label.detach().ne(self.tgt_pad)
        pred = logits.detach().max(2)[1]  # [batch_size, seq_len]
        num_correct = y_label.detach().eq(pred).float().masked_select(mask).sum() / normalization
        num_total = mask.sum().float()

        stats = Statistics(loss.item(), num_total, num_correct)

        return loss, stats


    def train_one_batch(self, batch):
        self.model.train()
        # self.model.zero_grad()
        src_words, tgt_words, src_lengths, tgt_lengths = self.pair_data_variable(batch)
        loss, stat = self.compute_forward(src_words, tgt_words, src_lengths)

        return stat

    def valid(self, global_step):
        valid_stat = Statistics()
        self.model.eval()
        for batch in create_batch_iter(self.valid_data, self.config.test_batch_size):
            src_words, tgt_words, src_lengths, tgt_lengths = self.pair_data_variable(batch)
            loss, stat = self.compute_forward(src_words, tgt_words, src_lengths)
            valid_stat.update(stat)
        valid_stat.print_valid(global_step)
        return valid_stat


    def translate(self, eval_data):
        self.model.eval()
        result = {}
        for batch in create_batch_iter(eval_data, self.config.test_batch_size):
            batch_size = len(batch)
            src_words, src_lengths = self.source_data_variable(batch)

            allHyp = self.translate_batch(src_words, src_lengths)
            all_hyp_inds = [beam_result[0] for beam_result in allHyp]
            # for idx in range(batch_size):
            #     if all_hyp_inds[idx][-1] == self.tgt_vocab.EOS:
            #         all_hyp_inds[idx].pop()

            all_hyp_words = []
            for idxs in all_hyp_inds:
                all_hyp_words += [[self.tgt_vocab.id2word(idx) for idx in idxs]]

            for idx, instance in enumerate(batch):
                result['\t'.join(instance[1])] = all_hyp_words[idx]

        return result


    def translate_batch(self, src_inputs, src_input_lengths):
        word_ids = self.model(src_inputs, lengths=src_input_lengths, mode="infer", beam_size=self.config.beam_size)
        # print(word_ids.size())

        word_ids = word_ids.cpu().numpy().tolist()
        result = []
        for sent_t in word_ids:
            sent_t = [[wid for wid in line if (wid != self.tgt_eos and wid != self.tgt_pad)] for line in sent_t]
            result.append(sent_t)

        return result
