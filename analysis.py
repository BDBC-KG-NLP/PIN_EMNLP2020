import json
import numpy as np
import collections

all_v = [0., 0.]
all_pred_v = [0., 0.]

def to_dict(sv):
    sv_dict = {}
    for s_v in sv:
        s, v = s_v.rsplit('-', 1)
        sv_dict[s] = v
    return sv_dict

# def check(self, sv_dict_true, sv_dict_pred, history):
#     sv_dict_true = self.to_dict(sv_dict_true)
#     sv_dict_pred = self.to_dict(sv_dict_pred)
#     output = [[], [], [], []]
#     all_s = set(sv_dict_true.keys()) | set(sv_dict_pred.keys())
#     for s in all_s:
#         if sv_dict_true[s] not in history:
#             if s not in sv_dict_pred or sv_dict_pred[s]!=sv_dict_true[s]:
#                 output[3].append('-'.join([s, sv_dict_true[s], sv_dict_pred[s]]))
#         if s in sv_dict_true and s not in sv_dict_pred:
#             output[0].append('-'.join([s, sv_dict_true[s]]))
#         elif s in sv_dict_true and s in sv_dict_pred:
#             if sv_dict_true[s]!=sv_dict_pred[s]:
#                 output[1].append('-'.join([s, sv_dict_true[s], sv_dict_pred[s]]))
#         elif s not in sv_dict_true and s in sv_dict_pred:
#             output[2].append('-'.join([s, sv_dict_pred[s]]))
#     return output

# def check(sv_dict_true, sv_dict_pred, history):
#     sv_dict_true = to_dict(sv_dict_true)
#     sv_dict_pred = to_dict(sv_dict_pred)
#     all_s = set(sv_dict_true.keys())
#     for s in all_s:
#         if sv_dict_true[s] not in history:
#             all_v[1] += 1.
#             if s in sv_dict_pred and sv_dict_pred[s]==sv_dict_true[s]:
#                 all_pred_v[1] += 1.            
#         else:
#             all_v[0] += 1.
#             if s in sv_dict_pred and sv_dict_pred[s]==sv_dict_true[s]:
#                 all_pred_v[0] += 1.

# with open('all_prediction_TRADE.json') as f:
#     data = json.load(f)
#     for dia in data.values():
#         for turn in dia.values():
#             check(turn['turn_belief'], turn['pred_bs_ptr'], turn['context_plain'])

# print all_v, all_pred_v

with open('all_prediction_TRADE.json') as f:
    data = json.load(f)
# sampled = np.random.choice(list(data.keys()), 100, replace=False)
# sample_dias = {'sampled': list(sampled)}
with open('sampled.json') as f:
    sampled=json.load(f)['sampled']
dias = {}
for dia in sampled:
    dia_content = {}
    for tid in data[dia]:
        dia_content[tid] = data[dia][tid]
        new_belief = [[x, [-1, -1, -1]] for x in dia_content[tid]['turn_belief']]
        dia_content[tid]['turn_belief'] = new_belief
    order_turn = sorted(dia_content.items(), key=lambda x: x[0])
    dias[dia] = collections.OrderedDict(order_turn)
with open('sampled_dias.json', 'wt') as f:
    json.dump(dias, f, indent=4)