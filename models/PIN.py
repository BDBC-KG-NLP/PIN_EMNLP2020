import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import random
import numpy as np

# import matplotlib.pyplot as plt
# import seaborn  as sns
# import nltk
import os
import json
# import pandas as pd
import copy

from utils.measures import wer, moses_multi_bleu
from utils.masked_cross_entropy import *
from utils.config import *
import pprint

class PIN(nn.Module):
    def __init__(self, hidden_size, lang, path, task, lr, dropout, slots, gating_dict, nb_train_vocab=0):
        super(PIN, self).__init__()
        self.name = "PIN"
        self.task = task
        self.hidden_size = hidden_size    
        self.lang = lang[0]
        self.mem_lang = lang[1] 
        self.lr = lr 
        self.dropout = dropout
        self.slots = slots[0]
        self.slot_temp = slots[2]
        self.gating_dict = gating_dict
        self.nb_gate = len(gating_dict)
        self.cross_entorpy = nn.CrossEntropyLoss()

        self.sys_encoder = EncoderRNN(self.lang.n_words, hidden_size, self.dropout)
        self.user_encoder = EncoderRNN(self.lang.n_words, hidden_size, self.dropout)
        self.Irnn_sys = InteractiveRNN(hidden_size)
        self.Irnn_user = InteractiveRNN(hidden_size)
        self.decoder = Generator(self.lang, self.sys_encoder.embedding, self.lang.n_words, hidden_size, self.dropout, self.slots, self.nb_gate) 
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)       
        
        if path:
            self.load_model(path)

        # Initialize criterion
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=1, min_lr=1e-4, verbose=True)
        
        self.reset()
        if USE_CUDA:
            self.cuda()

    def print_loss(self):    
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_gate = self.loss_gate / self.print_every
        print_loss_class = self.loss_class / self.print_every
        # print_loss_domain = self.loss_domain / self.print_every
        self.print_every += 1     
        return 'L:{:.2f},LP:{:.2f},LG:{:.2f}'.format(print_loss_avg,print_loss_ptr,print_loss_gate)
    
    def save_model(self, dec_type):
        directory = 'save/TRADE-'+args["addName"]+args['dataset']+str(self.task)+'/'+'HDD'+str(self.hidden_size)+'BSZ'+str(args['batch'])+'DR'+str(self.dropout)+str(dec_type)                 
        if not os.path.exists(directory):
            os.makedirs(directory)
        state = {
        'args': args,
        'model': self.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, directory + '/model.th')

    def load_model(self, path):
        state = torch.load(str(path)+'/model.th')
        self.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
    
    def reset(self):
        self.loss, self.print_every, self.loss_ptr, self.loss_gate, self.loss_class = 0, 1, 0, 0, 0

    def train_batch(self, data, clip, slot_temp, reset=0):
        if reset: self.reset()
        # Zero gradients of both optimizers
        self.optimizer.zero_grad()
        
        # Encode and Decode
        use_teacher_forcing = random.random() < args["teacher_forcing_ratio"]

        all_point_outputs, gates, words_point_out, words_class_out, all_generate_y, all_y_length, all_gating_label = self.encode_and_decode(data, use_teacher_forcing, slot_temp)

        loss_ptr = masked_cross_entropy_for_value(
            all_point_outputs.transpose(0, 1).contiguous(),
            all_generate_y.contiguous(), #[:,:len(self.point_slots)].contiguous(),
            all_y_length) #[:,:len(self.point_slots)])
        loss_gate = self.cross_entorpy(gates.transpose(0, 1).contiguous().view(-1, gates.size(-1)), all_gating_label.contiguous().view(-1))

        if args["use_gate"]:
            loss = loss_ptr + loss_gate
        else:
            loss = loss_ptr

        self.loss_grad = loss
        self.loss_ptr_to_bp = loss_ptr
        
        # Update parameters with optimizers
        self.loss += loss.data
        self.loss_ptr += loss_ptr.item()
        self.loss_gate += loss_gate.item()

        del all_point_outputs
        del gates
        del words_point_out
        del words_class_out
        del all_generate_y
        del all_y_length
        del all_gating_label
    
    def optimize(self, clip):
        self.loss_grad.backward()
        clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.optimizer.step()
        del self.loss_grad

    def optimize_GEM(self, clip):
        clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.optimizer.step()

    def rand_musk_seqs(self, seqs, dropout):
        story_size = seqs.size()
        rand_mask = np.ones(story_size)
        bi_mask = np.random.binomial([np.ones((story_size[0],story_size[1]))], 1-dropout)[0]
        rand_mask = rand_mask * bi_mask
        rand_mask = torch.Tensor(rand_mask)
        if USE_CUDA: 
            rand_mask = rand_mask.cuda()
        story = seqs * rand_mask.long()
        return story

    def encode_and_decode(self, data, use_teacher_forcing, slot_temp):

        # Encode dialog history
        batch_size = len(data['turn_count'])
        sys_hidden = None
        user_hidden = None
        sys_h = None
        user_h = None
        sys_H = []
        user_H = []
        story_sys = []
        story_user = []
        sys_musk = []
        user_musk = []

        # The Parallel Interactive RNN
        for i in range(data['turn_count'][0]):
            if args['unk_mask'] and self.training:
                story_s = self.rand_musk_seqs(data['turn_sys'][i], self.dropout)
                story_u = self.rand_musk_seqs(data['turn_user'][i], self.dropout)
            else:
                story_s = data['turn_sys'][i]
                story_u = data['turn_user'][i]
            story_sys.append(story_s)
            story_user.append(story_u) 
            sys_musk.append(data['turn_sys_musk'][i])
            user_musk.append(data['turn_user_musk'][i])       
            sys_output, sys_h = self.sys_encoder(story_s, data['turn_sys_len'][i], sys_hidden)
            user_output, user_h = self.user_encoder(story_u, data['turn_user_len'][i], user_hidden)
            user_out, sys_hidden = self.Irnn_sys(user_output, sys_h)
            sys_out, user_hidden = self.Irnn_user(sys_output, user_h)           
            sys_H.append(sys_out)
            user_H.append(user_out)


        max_res_len = data['generate_y'].size(2) if self.training else 10

        all_point_outputs, all_gate_outputs, words_point_out, words_class_out = self.decoder.forward(batch_size, torch.cat(sys_H, 1), torch.cat(
            user_H, 1), torch.cat(story_sys, 1), torch.cat(story_user, 1), torch.cat(sys_musk, 1), torch.cat(
            user_musk, 1), max_res_len, data['generate_y'], use_teacher_forcing, slot_temp)

        return  all_point_outputs, all_gate_outputs, words_point_out, words_class_out, data['generate_y'], data['y_lengths'], data['gating_label']

    def evaluate(self, dev, matric_best, slot_temp, early_stop=None):
        # Set to not-training mode to disable dropout
        self.train(False) 
        print("STARTING EVALUATION")
        all_prediction = {}
        inverse_unpoint_slot = dict([(v, k) for k, v in self.gating_dict.items()])
        pbar = tqdm(enumerate(dev),total=len(dev))
        for j, data_dev in pbar: 
            # Encode and Decode
            batch_size = len(data_dev['turn_count'])
            _, gates, words, class_words, generate_y, y_length, gating_label = self.encode_and_decode(data_dev, False, slot_temp)

            del _
            del class_words
            del generate_y
            del y_length
            del gating_label

            for bi in range(batch_size):
                if data_dev["ID"][bi] not in all_prediction.keys():
                    all_prediction[data_dev["ID"][bi]] = {}
                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]] = {"turn_belief":data_dev["turn_belief"][bi]}
                predict_belief_bsz_ptr, predict_belief_bsz_class = [], []
                gate = torch.argmax(gates.transpose(0, 1)[bi], dim=1)

                # pointer-generator results
                if args["use_gate"]:
                    for si, sg in enumerate(gate):
                        if sg==self.gating_dict["none"]:
                            continue
                        elif sg==self.gating_dict["ptr"]:
                            pred = np.transpose(words[si])[bi]
                            st = []
                            for e in pred:
                                if e== 'EOS': break
                                else: st.append(e)
                            st = " ".join(st)
                            if st == "none":
                                continue
                            else:
                                predict_belief_bsz_ptr.append(slot_temp[si]+"-"+str(st))
                        else:
                            predict_belief_bsz_ptr.append(slot_temp[si]+"-"+inverse_unpoint_slot[sg.item()])
                else:
                    for si, _ in enumerate(gate):
                        pred = np.transpose(words[si])[bi]
                        st = []
                        for e in pred:
                            if e == 'EOS': break
                            else: st.append(e)
                        st = " ".join(st)
                        if st == "none":
                            continue
                        else:
                            predict_belief_bsz_ptr.append(slot_temp[si]+"-"+str(st))

                turn_label = []
                for tl in data_dev["turn_label"][bi]:
                    turn_label.append('-'.join(tl))

                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]]["pred_bs_ptr"] = predict_belief_bsz_ptr
                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]]["dialog_history"] = data_dev["dialog_history"][bi]
                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]]["turn_label"] = turn_label
                         
            del gates
            del words

        if args["genSample"]:
            json.dump(all_prediction, open("all_prediction_{}.json".format(self.name), 'w'), indent=4)

        joint_acc_score_ptr, F1_score_ptr, turn_acc_score_ptr = self.evaluate_metrics(all_prediction, "pred_bs_ptr", slot_temp)

        evaluation_metrics = {"Joint Acc":joint_acc_score_ptr, "Turn Acc":turn_acc_score_ptr, "Joint F1":F1_score_ptr}
        print(evaluation_metrics)
        json.dump(evaluation_metrics, open("evaluation_metrics_{}.json".format(self.name), 'w'), indent=4)
        
        # Set back to training mode
        self.train(True)

        joint_acc_score = joint_acc_score_ptr # (joint_acc_score_ptr + joint_acc_score_class)/2
        F1_score = F1_score_ptr

        if (early_stop == 'F1'):
            if (F1_score >= matric_best):
                self.save_model('ENTF1-{:.4f}'.format(F1_score))
                print("MODEL SAVED")  
            return F1_score
        else:
            if (joint_acc_score >= matric_best):
                self.save_model('ACC-{:.4f}'.format(joint_acc_score))
                print("MODEL SAVED")
            return joint_acc_score

    def to_dict(self, sv):
        sv_dict = {}
        for s_v in sv:
            s, v = s_v.rsplit('-', 1)
            sv_dict[s] = v
        return sv_dict

    def check(self, sv_dict_true, sv_dict_pred):
        sv_dict_true = self.to_dict(sv_dict_true)
        sv_dict_pred = self.to_dict(sv_dict_pred)
        output = [[], [], []]
        all_s = set(sv_dict_true.keys()) | set(sv_dict_pred.keys())
        for s in all_s:
            if s in sv_dict_true and s not in sv_dict_pred:
                output[0].append('-'.join([s, sv_dict_true[s]]))
            elif s in sv_dict_true and s in sv_dict_pred:
                if sv_dict_true[s]!=sv_dict_pred[s]:
                    output[1].append('-'.join([s, sv_dict_true[s], sv_dict_pred[s]]))
            elif s not in sv_dict_true and s in sv_dict_pred:
                output[2].append('-'.join([s, sv_dict_pred[s]]))
        return output

    def evaluate_metrics(self, all_prediction, from_which, slot_temp):
        total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
        for d, v in all_prediction.items():
            for t in range(len(v)):
                cv = v[t]
                if set(cv["turn_belief"]) == set(cv[from_which]):
                    joint_acc += 1
                else:
                    self.check(cv["turn_belief"], cv[from_which])
                total += 1

                # Compute prediction slot accuracy
                temp_acc = self.compute_acc(set(cv["turn_belief"]), set(cv[from_which]), slot_temp)
                turn_acc += temp_acc

                # Compute prediction joint F1 score
                temp_f1, temp_r, temp_p, count = self.compute_prf(set(cv["turn_belief"]), set(cv[from_which]))
                F1_pred += temp_f1
                F1_count += count

        joint_acc_score = joint_acc / float(total) if total!=0 else 0
        turn_acc_score = turn_acc / float(total) if total!=0 else 0
        F1_score = F1_pred / float(F1_count) if F1_count!=0 else 0
        return joint_acc_score, F1_score, turn_acc_score

    def compute_acc(self, gold, pred, slot_temp):
        miss_gold = 0
        miss_slot = []
        for g in gold:
            if g not in pred:
                miss_gold += 1
                miss_slot.append(g.rsplit("-", 1)[0])
        wrong_pred = 0
        for p in pred:
            if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
                wrong_pred += 1
        ACC_TOTAL = len(slot_temp)
        ACC = len(slot_temp) - miss_gold - wrong_pred
        ACC = ACC / float(ACC_TOTAL)
        return ACC

    def compute_prf(self, gold, pred):
        TP, FP, FN = 0, 0, 0
        if len(gold)!= 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in pred:
                if p not in gold:
                    FP += 1
            precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
            recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
        else:
            if len(pred)==0:
                precision, recall, F1, count = 1, 1, 1, 1
            else:
                precision, recall, F1, count = 0, 0, 0, 1
        return F1, recall, precision, count


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout, n_layers=1):
        super(EncoderRNN, self).__init__()      
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size  
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_token)
        self.embedding.weight.data.normal_(0, 0.1)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        # self.domain_W = nn.Linear(hidden_size, nb_domain)

        if args["load_embedding"]:
            with open(os.path.join("data/", 'emb{}.json'.format(vocab_size))) as f:
                E = json.load(f)
            new = self.embedding.weight.data.new
            self.embedding.weight.data.copy_(new(E))
            self.embedding.weight.requires_grad = True
            print("Encoder embedding requires_grad", self.embedding.weight.requires_grad)

        if args["fix_embedding"]:
            self.embedding.weight.requires_grad = False

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        embedded = self.dropout_layer(embedded) 
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths:
           outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)   
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]
        return outputs, hidden


class InteractiveRNN(nn.Module):
    def __init__(self, hidden_size):
        super(InteractiveRNN, self).__init__()
        self.hidden_size = hidden_size 
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, input_x, input_y):
        outputs, hidden = self.gru(input_x, input_y)
        outputs = outputs[:,:,:self.hidden_size] + outputs[:,:,self.hidden_size:]
        return outputs, hidden      

class Generator(nn.Module):
    def __init__(self, lang, shared_emb, vocab_size, hidden_size, dropout, slots, nb_gate):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.lang = lang
        self.embedding = shared_emb 
        self.dropout_layer = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout)
        self.nb_gate = nb_gate
        self.hidden_size = hidden_size
        self.W_c = nn.Linear(hidden_size, 1)
        self.W_copy = nn.Linear(3*hidden_size, 1)
        self.W_ratio = nn.Linear(4*hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.slots = slots
        self.slot_att = nn.ParameterDict({slot: nn.Parameter(torch.randn(1, hidden_size).normal_(0, 0.1)) for slot in self.slots})

        self.W_gate = nn.Linear(2*hidden_size, nb_gate)

        # Create independent slot embeddings
        self.slot_w2i = {}
        for slot in self.slots:
            if slot.split("-")[0] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[0]] = len(self.slot_w2i)
            if slot.split("-")[1] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[1]] = len(self.slot_w2i)
        self.Slot_emb = nn.Embedding(len(self.slot_w2i), hidden_size)
        self.Slot_emb.weight.data.normal_(0, 0.1)

    def forward(self, batch_size, sys_H, user_H, story_sys, story_user, sys_musk, user_musk, max_res_len, target_batches, use_teacher_forcing, slot_temp):
        all_point_outputs = torch.zeros(len(slot_temp), batch_size, max_res_len, self.vocab_size)
        all_gate_outputs = torch.zeros(len(slot_temp), batch_size, self.nb_gate)
        if USE_CUDA: 
            all_point_outputs = all_point_outputs.cuda()
            all_gate_outputs = all_gate_outputs.cuda()
        
        # Get the slot embedding 
        slot_emb_dict = {}
        for slot in slot_temp:
            # Domain embbeding
            if slot.split("-")[0] in self.slot_w2i.keys():
                domain_w2idx = [self.slot_w2i[slot.split("-")[0]]]
                domain_w2idx = torch.tensor(domain_w2idx)
                if USE_CUDA: domain_w2idx = domain_w2idx.cuda()
                domain_emb = self.Slot_emb(domain_w2idx)
            # Slot embbeding
            if slot.split("-")[1] in self.slot_w2i.keys():
                slot_w2idx = [self.slot_w2i[slot.split("-")[1]]]
                slot_w2idx = torch.tensor(slot_w2idx)
                if USE_CUDA: slot_w2idx = slot_w2idx.cuda()
                slot_emb = self.Slot_emb(slot_w2idx)

            # Combine two embeddings as one query
            slot_emb_dict[slot] = domain_emb + slot_emb

        words_class_out = []
        # Compute pointer-generator output
        words_point_out = []
        counter = 0  

        for slot in slot_temp:

            # Build slot-level context
            hidden_sys, _, _ = self.attend(sys_H, self.slot_att[slot].expand(batch_size, self.hidden_size), sys_musk)
            hidden_user, _, _ = self.attend(user_H, self.slot_att[slot].expand(batch_size, self.hidden_size), user_musk)
            hidden = torch.unsqueeze(hidden_sys+hidden_user, 0)

            # Decode
            words = []
            slot_emb = slot_emb_dict[slot]
            decoder_input = self.dropout_layer(slot_emb).expand(batch_size, self.hidden_size)
            for wi in range(max_res_len):
                dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)
                context_sys, logits_sys, prob_sys = self.attend(sys_H, hidden.squeeze(0), sys_musk)
                context_user, logits_user, prob_user = self.attend(user_H, hidden.squeeze(0), user_musk)
                M = torch.cat([context_sys, context_user], -1)
                if wi == 0: 
                    all_gate_outputs[counter] = self.W_gate(M)

                # Distributed copy
                p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))
                p_gen_vec = torch.cat([dec_state.squeeze(0), M, decoder_input], -1)
                vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))
                p_context_sys = torch.zeros(p_vocab.size())
                p_context_user = torch.zeros(p_vocab.size())
                if USE_CUDA: 
                    p_context_sys = p_context_sys.cuda()
                    p_context_user = p_context_user.cuda()
                p_context_sys.scatter_add_(1, story_sys, prob_sys)
                p_context_user.scatter_add_(1, story_user, prob_user)
                p_w = F.softmax(torch.stack([self.W_copy(torch.cat([dec_state.squeeze(0), context_sys, decoder_input], -1)), self.W_copy(torch.cat([dec_state.squeeze(0), context_user, decoder_input], -1))], 1), 1).squeeze(2)
                alpha, beta = torch.split(p_w, 1, 1)
                final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_vocab) * (alpha.expand_as(p_vocab)*p_context_sys+beta.expand_as(p_vocab)*p_context_user) + \
                                vocab_pointer_switches.expand_as(p_vocab) * p_vocab
                pred_word = torch.argmax(final_p_vocab, dim=1)
                words.append([self.lang.index2word[w_idx.item()] for w_idx in pred_word])
                all_point_outputs[counter, :, wi, :] = final_p_vocab
                if use_teacher_forcing:
                    decoder_input = self.embedding(target_batches[:, counter, wi]) # Chosen word is next input
                else:
                    decoder_input = self.embedding(pred_word)   
                if USE_CUDA: decoder_input = decoder_input.cuda()
            counter += 1
            words_point_out.append(words)
        return all_point_outputs, all_gate_outputs, words_point_out, words_class_out

    def attend(self, seq, cond, musk=None):
        """
        attend over the sequences `seq` using the condition `cond`.
        """
        scores_ = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
        if musk is not None:
            scores_ = scores_+musk
        scores = F.softmax(scores_, dim=1)
        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context, scores_, scores


    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1,0))
        scores = F.softmax(scores_, dim=1)
        return scores


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
