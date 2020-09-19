# encoding=utf8
import sys
sys.path.append('../')

# reload(sys)
# sys.setdefaultencoding('utf8')

import json
import torch
import torch.utils.data as data
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
import ast
from collections import Counter
from collections import OrderedDict
# from trade.embedding import ComposedEmbedding
from tqdm import tqdm
import os
import numpy as np
import pickle
from random import shuffle

from .fix_label import *

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
        self.n_words = len(self.index2word) # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])
      
    def index_words(self, sent, type):
        if type == 'utter':
            for word in sent.split():
                self.index_word(word)
        elif type == 'slot':
            for slot in sent:
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split():
                    self.index_word(ss)
        elif type == 'belief':
            for slot, value in sent.items():
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split():
                    self.index_word(ss)
                for v in value.split():
                    self.index_word(v)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_info, src_word2id, trg_word2id, sequicity, mem_word2id):
        """Reads source and target sequences from txt files."""
        self.data = data_info
        self.sequicity = sequicity
        # self.num_total_dias = len(data_info)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.mem_word2id = mem_word2id
    
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        samples = []
        for i in index:
            turn = self.data[i]
            ID = turn['ID']
            turn_id = turn['turn_id']
            turn_count = turn['turn_count']
            turn_label = turn['turn_label']
            turn_belief = turn['turn_belief']
            gating_label = turn['gating_label']
            dialog_history = turn['dialog_history']
            turn_domain = self.preprocess_domain(turn['turn_domain'])
            generate_y = turn['generate_y']
            generate_y = self.preprocess_slot(generate_y, self.trg_word2id)
            turn_sys = [self.preprocess(sys, self.src_word2id) for sys in turn['turn_sys']] 
            turn_user = [self.preprocess(user, self.src_word2id) for user in turn['turn_user']]
        
            item_info = {
                "ID":ID, 
                "turn_id":turn_id, 
                "turn_count":turn_count, 
                "turn_label": turn_label,
                "turn_belief":turn_belief, 
                "gating_label":gating_label, 
                "turn_sys":turn_sys, 
                "turn_user":turn_user,  
                "dialog_history":dialog_history,                           
                "turn_domain":turn_domain, 
                "generate_y":generate_y, 
                }

            samples.append(item_info)

        return samples

    # def __len__(self):
    #     return self.num_total_dias
    
    def preprocess(self, sequence, word2idx):
        """Converts words to ids."""
        story = [word2idx[word] if word in word2idx else UNK_token for word in sequence.split()]
        if len(story)==0:
            story.append(PAD_token)          
        story = torch.Tensor(story)      
        return story

    def preprocess_slot(self, sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            v = [word2idx[word] if word in word2idx else UNK_token for word in value.split()] + [EOS_token]
            story.append(v)           
        # story = torch.Tensor(story)
        return story

    def preprocess_memory(self, sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            d, s, v = value
            s = s.replace("book","").strip()
            # separate each word in value to different memory slot
            for wi, vw in enumerate(v.split()):
                idx = [word2idx[word] if word in word2idx else UNK_token for word in [d, s, "t{}".format(wi), vw]]
                story.append(idx)
        if len(story)==0:
            story.append(PAD_token)
        story = torch.Tensor(story)
        return story

    def preprocess_domain(self, turn_domain):
        domains = {"attraction":0, "restaurant":1, "taxi":2, "train":3, "hotel":4, "hospital":5, "bus":6, "police":7}
        return domains[turn_domain]

def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        padded_seqs = torch.ones(len(sequences), max_len).long()
        musk_seqs = -np.inf*torch.ones(len(sequences), max_len)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
            musk_seqs[i, :end] = torch.Tensor([0. for x in range(end)])
        padded_seqs = padded_seqs.detach() #torch.tensor(padded_seqs)
        return padded_seqs, lengths, musk_seqs

    def merge_multi_response(sequences):
        '''
        merge from batch * nb_slot * slot_len to batch * nb_slot * max_slot_len
        '''
        lengths = []
        for bsz_seq in sequences:
            length = [len(v) for v in bsz_seq]
            lengths.append(length)
        max_len = max([max(l) for l in lengths])
        padded_seqs = []
        for bsz_seq in sequences:
            pad_seq = []
            for v in bsz_seq:
                v = v + [PAD_token] * (max_len-len(v))
                pad_seq.append(v)
            padded_seqs.append(pad_seq)
        padded_seqs = torch.tensor(padded_seqs)
        lengths = torch.tensor(lengths)
        return padded_seqs, lengths

    data = data[0]

    keys = data[0].keys()
    turns = data[0]['turn_count']
    turn = {}
  
    for key in keys:
        if key!='turn_sys' and key!='turn_user':
            turn[key] = [d[key] for d in data]
        else:
            turn[key] = [[d[key][i] for d in data] for i in range(turns)]

    y_seqs, y_lengths = merge_multi_response(turn["generate_y"])
    gating_label = torch.tensor(turn["gating_label"])
    turn_domain = torch.tensor(turn["turn_domain"])            

    sys_seqs, sys_lengths, sys_musks, user_seqs, user_lengths, user_musks = [], [], [], [], [], []
    for i in range(turns):
        sys_seq, sys_length, sys_musk = merge(turn['turn_sys'][i])
        user_seq, user_length, user_musk = merge(turn['turn_user'][i])
        sys_seqs.append(sys_seq)
        sys_lengths.append(sys_length)
        sys_musks.append(sys_musk)
        user_seqs.append(user_seq)
        user_lengths.append(user_length)
        user_musks.append(user_musk)


    if USE_CUDA:
        gating_label = gating_label.cuda()
        turn_domain = turn_domain.cuda()
        y_seqs = y_seqs.cuda()
        y_lengths = y_lengths.cuda()
        for i in range(turns):
            sys_seqs[i] = sys_seqs[i].cuda()
            sys_musks[i] = sys_musks[i].cuda()
            user_seqs[i] = user_seqs[i].cuda()
            user_musks[i] = user_musks[i].cuda()

    turn['turn_sys'] = sys_seqs
    turn['turn_sys_len'] = sys_lengths
    turn['turn_sys_musk'] = sys_musks
    turn['turn_user'] = user_seqs
    turn['turn_user_len'] = user_lengths
    turn['turn_user_musk'] = user_musks
    turn['gating_label'] = gating_label
    turn['turn_domain'] = turn_domain
    turn['generate_y'] = y_seqs
    turn['y_lengths'] = y_lengths

    return turn

def read_langs(file_name, gating_dict, SLOTS, dataset, lang, mem_lang, sequicity, training, max_line = None):
    print(("Reading from {}".format(file_name)))
    data = []
    max_resp_len, max_value_len = 0, 0
    domain_counter = {} 
    with open(file_name) as f:
        dials = json.load(f)
        # create vocab first 
        for dial_dict in dials:
            if (args["all_vocab"] or dataset=="train") and training:
                for ti, turn in enumerate(dial_dict["dialogue"]):
                    lang.index_words(turn["system_transcript"], 'utter')
                    lang.index_words(turn["transcript"], 'utter')
        # determine training data ratio, default is 100%
        if training and dataset=="train" and args["data_ratio"]!=100:
            random.Random(10).shuffle(dials)
            dials = dials[:int(len(dials)*0.01*args["data_ratio"])]
        
        cnt_lin = 1
        for dial_dict in dials:
            dialog_history = ""            
            last_belief_dict = {}
            # Filtering and counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Unseen domain setting
            if args["only_domain"] != "" and args["only_domain"] not in dial_dict["domains"]:
                continue
            if (args["except_domain"] != "" and dataset == "test" and args["except_domain"] not in dial_dict["domains"]) or \
               (args["except_domain"] != "" and dataset != "test" and [args["except_domain"]] == dial_dict["domains"]): 
                continue

            # Reading data
            turn_sys = []
            turn_user = []
            for ti, turn in enumerate(dial_dict["dialogue"]):
                turn_domain = turn["domain"]
                turn_id = turn["turn_idx"]
                turn_count = ti+1
                turn_label = turn['turn_label']
                turn_sys.append(turn["system_transcript"].strip())
                turn_user.append(turn["transcript"].strip())
                dialog_history +=  (turn["system_transcript"] + " ; " + turn["transcript"] + " ; ")
                source_text = dialog_history.strip()                
                turn_belief_dict = fix_general_label_error(turn["belief_state"], False, SLOTS)

                # Generate domain-dependent slot list
                slot_temp = SLOTS
                if dataset == "train" or dataset == "dev":
                    if args["except_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["except_domain"] not in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["except_domain"] not in k])
                    elif args["only_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["only_domain"] in k])
                else:
                    if args["except_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["except_domain"] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["except_domain"] in k])
                    elif args["only_domain"] != "":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["only_domain"] in k])

                turn_belief_list = [str(k)+'-'+str(v) for k, v in turn_belief_dict.items()]

                if (args["all_vocab"] or dataset=="train") and training:
                    mem_lang.index_words(turn_belief_dict, 'belief')

                class_label, generate_y, slot_mask, gating_label  = [], [], [], []
                start_ptr_label, end_ptr_label = [], []
                for slot in slot_temp:
                    if slot in turn_belief_dict.keys(): 
                        generate_y.append(turn_belief_dict[slot])

                        if turn_belief_dict[slot] == "dontcare":
                            gating_label.append(gating_dict["dontcare"])
                        elif turn_belief_dict[slot] == "none":
                            gating_label.append(gating_dict["none"])
                        else:
                            gating_label.append(gating_dict["ptr"])

                        if max_value_len < len(turn_belief_dict[slot]):
                            max_value_len = len(turn_belief_dict[slot])

                    else:
                        generate_y.append("none")
                        gating_label.append(gating_dict["none"])
                
                data_detail = {
                    "ID":dial_dict["dialogue_idx"], 
                    "domains":dial_dict["domains"], 
                    "turn_domain":turn_domain,
                    "turn_id":turn_id, 
                    "turn_count":turn_count,
                    "turn_label": turn_label,
                    "turn_sys":turn_sys, 
                    "turn_user":turn_user,
                    "dialog_history":source_text,                     
                    "turn_belief":turn_belief_list,
                    "gating_label":gating_label, 
                    'generate_y':generate_y
                    }
                data.append(data_detail)
                
                if max_resp_len < len(source_text.split()):
                    max_resp_len = len(source_text.split())

            cnt_lin += 1
            if(max_line and cnt_lin>=max_line):
                break

    # add t{} to the lang file
    if "t{}".format(max_value_len-1) not in mem_lang.word2index.keys() and training:
        for time_i in range(max_value_len):
            mem_lang.index_words("t{}".format(time_i), 'utter')

    print("domain_counter", domain_counter)
    return data, max_resp_len, slot_temp


def get_seq(pairs, lang, mem_lang, batch_size, type, sequicity): 
    if(type and args['fisher_sample']>0):
        shuffle(pairs)
        pairs = pairs[:args['fisher_sample']]

    dataset = Dataset(pairs, lang.word2index, lang.word2index, sequicity, mem_lang.word2index)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=1,
                                              # shuffle=type,
                                              collate_fn=collate_fn,
                                              sampler=ImbalancedDatasetSampler(dataset, batch_size))
    return data_loader


def dump_pretrained_emb(word2index, index2word, dump_path):
    print("Dumping pretrained embeddings...")
    embeddings = ComposedEmbedding()
    E = []
    for i in tqdm(range(len(word2index.keys()))):
        w = index2word[i]
        e = embeddings.lookup(w)
        E.append(e)
    with open(dump_path, 'wt') as f:
        json.dump(E, f)


def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ","").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    return SLOTS


def prepare_data_seq(training, task="dst", sequicity=0, batch_size=100):
    eval_batch = args["eval_batch"] if args["eval_batch"] else batch_size
    file_train = 'data/train_dials.json'
    file_dev = 'data/dev_dials.json'
    file_test = 'data/test_dials.json'
    # Create saving folder
    if args['path']:
        folder_name = args['path'].rsplit('/', 1)[0] + '/'
    else:
        folder_name = 'save/{}-'.format(args["decoder"])+args["addName"]+args['dataset']+str(args['task'])+'/'
    print("folder_name", folder_name)
    if not os.path.exists(folder_name): 
        os.makedirs(folder_name)
    # load domain-slot pairs from ontology
    ontology = json.load(open("data/multi-woz/MULTIWOZ2 2/ontology.json", 'r'))
    ALL_SLOTS = get_slot_information(ontology)
    gating_dict = {"ptr":0, "dontcare":1, "none":2}
    # Vocabulary
    lang, mem_lang = Lang(), Lang()
    lang.index_words(ALL_SLOTS, 'slot')
    mem_lang.index_words(ALL_SLOTS, 'slot')
    lang_name = 'lang-all.pkl' if args["all_vocab"] else 'lang-train.pkl'
    mem_lang_name = 'mem-lang-all.pkl' if args["all_vocab"] else 'mem-lang-train.pkl'

    if training:
        pair_train, train_max_len, slot_train = read_langs(file_train, gating_dict, ALL_SLOTS, "train", lang, mem_lang, sequicity, training)
        train = get_seq(pair_train, lang, mem_lang, batch_size, True, sequicity)
        nb_train_vocab = lang.n_words
        pair_dev, dev_max_len, slot_dev = read_langs(file_dev, gating_dict, ALL_SLOTS, "dev", lang, mem_lang, sequicity, training)
        dev   = get_seq(pair_dev, lang, mem_lang, eval_batch, False, sequicity)
        pair_test, test_max_len, slot_test = read_langs(file_test, gating_dict, ALL_SLOTS, "test", lang, mem_lang, sequicity, training)
        test  = get_seq(pair_test, lang, mem_lang, eval_batch, False, sequicity)
        if os.path.exists(folder_name+lang_name) and os.path.exists(folder_name+mem_lang_name):
            print("[Info] Loading saved lang files...")
            with open(folder_name+lang_name, 'rb') as handle: 
                lang = pickle.load(handle)
            with open(folder_name+mem_lang_name, 'rb') as handle: 
                mem_lang = pickle.load(handle)
        else:
            print("[Info] Dumping lang files...")
            with open(folder_name+lang_name, 'wb') as handle: 
                pickle.dump(lang, handle)
            with open(folder_name+mem_lang_name, 'wb') as handle: 
                pickle.dump(mem_lang, handle)
        emb_dump_path = 'data/emb{}.json'.format(len(lang.index2word))
        if not os.path.exists(emb_dump_path) and args["load_embedding"]:
            dump_pretrained_emb(lang.word2index, lang.index2word, emb_dump_path)
    else:
        with open(folder_name+lang_name, 'rb') as handle:
            lang = pickle.load(handle)
        with open(folder_name+mem_lang_name, 'rb') as handle:
            mem_lang = pickle.load(handle)

        pair_train, train_max_len, slot_train, train, nb_train_vocab = [], 0, {}, [], 0
        pair_dev, dev_max_len, slot_dev = read_langs(file_dev, gating_dict, ALL_SLOTS, "dev", lang, mem_lang, sequicity, training)
        dev   = get_seq(pair_dev, lang, mem_lang, eval_batch, False, sequicity)
        pair_test, test_max_len, slot_test = read_langs(file_test, gating_dict, ALL_SLOTS, "test", lang, mem_lang, sequicity, training)
        test  = get_seq(pair_test, lang, mem_lang, eval_batch, False, sequicity)

    test_4d = []
    if args['except_domain']!="":
        pair_test_4d, _, _ = read_langs(file_test, gating_dict, ALL_SLOTS, "dev", lang, mem_lang, sequicity, training)
        test_4d  = get_seq(pair_test_4d, lang, mem_lang, eval_batch, False, sequicity)

    max_word = max(train_max_len, dev_max_len, test_max_len) + 1

    print("Read %s pairs train" % len(pair_train))
    print("Read %s pairs dev" % len(pair_dev))
    print("Read %s pairs test" % len(pair_test))  
    print("Vocab_size: %s " % lang.n_words)
    print("Vocab_size Training %s" % nb_train_vocab )
    print("Vocab_size Belief %s" % mem_lang.n_words )
    print("Max. length of dialog words for RNN: %s " % max_word)
    print("USE_CUDA={}".format(USE_CUDA))

    SLOTS_LIST = [ALL_SLOTS, slot_train, slot_dev, slot_test]
    print("[Train Set & Dev Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST[2]))))
    print(SLOTS_LIST[2])
    print("[Test Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST[3]))))
    print(SLOTS_LIST[3])
    LANG = [lang, mem_lang]
    return train, dev, test, test_4d, LANG, SLOTS_LIST, gating_dict, nb_train_vocab

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dia_group = {}
        self.indices = []
        for i, dia in enumerate(dataset.data):
            if dia['turn_count'] not in self.dia_group:
                self.dia_group[dia['turn_count']] = []
            self.dia_group[dia['turn_count']].append(i)
        for g in self.dia_group:
            shuffle(self.dia_group[g])
            pointer = 0
            lens = len(self.dia_group[g])
            while pointer<lens:
                if pointer+batch_size<lens:
                    self.indices.append(self.dia_group[g][pointer:pointer+batch_size])
                    pointer = pointer+batch_size
                else:
                    self.indices.append(self.dia_group[g][pointer:])
                    pointer = lens   
        shuffle(self.indices)  
        self.len_data = len(self.indices)              
                
    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.len_data

