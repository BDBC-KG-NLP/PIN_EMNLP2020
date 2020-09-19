from tqdm import tqdm
import torch
import torch.nn as nn

from utils.config import *
from models.PIN import *

'''
python myTrain.py -dec= -bsz= -hdd= -dr= -lr=
'''

early_stop = args['earlyStop']

if args['dataset']=='multiwoz':
    from utils.utils_multiWOZ_DST import *
    early_stop = None
else:
    print("You need to provide the --dataset information")
    exit(1)

# Configure models and load data
avg_best, cnt, acc = 0.0, 0, 0.0
train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq(True, args['task'], False, batch_size=int(args['batch']))

model = globals()[args['decoder']](
    hidden_size=int(args['hidden']), 
    lang=lang, 
    path=args['path'], 
    task=args['task'], 
    lr=float(args['learn']), 
    dropout=float(args['drop']),
    slots=SLOTS_LIST,
    gating_dict=gating_dict, 
    nb_train_vocab=max_word)

# print("[Info] Slots include ", SLOTS_LIST)
# print("[Info] Unpointable Slots include ", gating_dict)

for epoch in range(50):
    print("Epoch:{}".format(epoch))  
    # Run the train function
    pbar = tqdm(enumerate(train),total=len(train))
    for i, data in pbar:
        model.train_batch(data, int(args['clip']), SLOTS_LIST[1], reset=(i==0))
        model.optimize(args['clip'])
        pbar.set_description(model.print_loss())

    if((epoch+1) % int(args['evalp']) == 0):
        
        acc = model.evaluate(dev, avg_best, SLOTS_LIST[2], early_stop)
        model.scheduler.step(acc)
        with open('train_records.txt', 'ab+') as f:
            f.write('   '.join([str(epoch), str(acc)])+'\n')        

        if(acc >= avg_best):
            avg_best = acc
            cnt=0
            best_model = model
        else:
            cnt+=1

        if(cnt == args["patience"] or (acc==1.0 and early_stop==None)): 
            print("Ran out of patient, early stop...")  
            break 

