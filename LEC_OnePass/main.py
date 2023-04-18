import tqdm
import time
import datetime
from datetime import datetime 
import random
#from document_reader import *
#from matres_reader import *
from matres_reader_with_tense import *
import os
from os import listdir
from os.path import isfile, join
from torch.utils.data import DataLoader
from util import *
from pprint import pprint
from transformers import AutoTokenizer, AutoModel
from model import transformers_mlp_cons
from exp import *
import numpy as np
import json
import sys

# datetime object containing current date and time
now = datetime.datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
print("date and time =", dt_string)

mask_in_input_ids = False
mask_in_input_mask = False
tense_acron = 1 # tense acronym (pastsimp): 1; else (past simple): 0

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
        
def docTransformerTokenIDs(sentences):
    if len(sentences) < 1:
        return None
    elif len(sentences) == 1:
        return sentences[0]['_subword_to_ID']
    else:
        TokenIDs = sentences[0]['_subword_to_ID']
        for i in range(1, len(sentences)):
            TokenIDs += sentences[i]['_subword_to_ID'][1:]
        return TokenIDs
    
def collate_fn(batch):
    max_len = max([len(f['input_ids']) for f in batch])
    input_ids = [f['input_ids'] + [0] * (max_len - len(f['input_ids'])) for f in batch]
    if mask_in_input_ids:
        for f_id, f in enumerate(input_ids):
            for event_p in batch[f_id]['event_pos']:
                f[event_p] = 67
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = [[1.0] * len(f['input_ids']) + [0.0] * (max_len - len(f['input_ids'])) for f in batch]
    if mask_in_input_mask:
        for f_id, f in enumerate(input_ids):
            for event_p in batch[f_id]['event_pos']:
                f[event_p] = 0.0
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    event_pos = [f['event_pos'] for f in batch]
    event_pos_end = [f['event_pos_end'] for f in batch]
    event_pair = [f['event_pair'] for f in batch]
    labels = [f['labels'] for f in batch]
    output = (input_ids, input_mask, event_pos, event_pos_end, event_pair, labels)
    return output

#############################
### Setting up parameters ###
#############################

params = {'transformers_model': 'google/bigbird-roberta-base',
          'dataset': sys.argv[6],   # 'HiEve', 'IC', 'MATRES' 
          'block_size': 64,
          'add_loss': float(sys.argv[5]), 
          'batch_size': int(sys.argv[3]),    # 6 works on 48G gpu
          'epochs': 30,
          'learning_rate': float(sys.argv[2]),    # subject to change
          'seed': 42,
          'gpu_id': sys.argv[4],    # subject to change
          'debug': 0,
          'rst_file_name': sys.argv[1],    # subject to change
          'mask_in_input_ids': mask_in_input_ids,
          'mask_in_input_mask': mask_in_input_mask,
          'marker': sys.argv[7], 
          'tense_acron': tense_acron,
         }
if params['transformers_model'][-5:] == "large":
    params['emb_size'] = 1024
elif params['transformers_model'][-4:] == "base":
    params['emb_size'] = 768
else:
    print("Something weird happens...")
    
set_seed(params['seed'])
rst_file_name = params['rst_file_name']
model_params_dir = "./model_params/"
if params['dataset'] == 'HiEve':
    best_PATH = model_params_dir + "HiEve_best/" + rst_file_name.replace(".rst", ".pt") # to save model params here
elif params['dataset'] == 'IC':
    best_PATH = model_params_dir + "IC_best/" + rst_file_name.replace(".rst", ".pt") # to save model params here
elif params['dataset'] == 'MATRES':
    best_PATH = model_params_dir + "MATRES_best/" + rst_file_name.replace(".rst", ".pt") # to save model params here
model_name = rst_file_name.replace(".rst", "")
with open("config/" + rst_file_name.replace("rst", "json"), 'w') as config_file:
    json.dump(params, config_file)
    
os.environ["CUDA_VISIBLE_DEVICES"] = params['gpu_id']
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
cuda = torch.device('cuda')
params['cuda'] = cuda # not included in config file

#######################
### Data processing ###
#######################
print("Processing " + params['dataset'] + " dataset...")
t0 = time.time()
if params['dataset'] == "IC":
    dir_name = "./IC/IC_Processed/"
    #max_sent_len = 193
elif params['dataset'] == "HiEve":
    dir_name = "./hievents_v2/processed/"
    #max_sent_len = 155
elif params['dataset'] == "MATRES":
    dir_name = ""
else:
    print("Not supporting this dataset yet!")
    
tokenizer = AutoTokenizer.from_pretrained(params['transformers_model'])   
if tense_acron:
    spec_toke_list = []
    special_tokens_dict = {'additional_special_tokens': ['[futuperfsimp]','[futucont]','[futuperfcont]','[futusimp]', '[pastcont]', '[pastperfcont]', '[pastperfsimp]', '[pastsimp]', '[prescont]', '[presperfcont]', '[presperfsimp]', '[pressimp]', '[futuperfsimppass]','[futucontpass]','[futuperfcontpass]','[futusimppass]', '[pastcontpass]', '[pastperfcontpass]', '[pastperfsimppass]', '[pastsimppass]', '[prescontpass]', '[presperfcontpass]', '[presperfsimppass]', '[pressimppass]', '[none]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model = AutoModel.from_pretrained(params['transformers_model'])
    model.resize_token_embeddings(len(tokenizer))
else:
    model = AutoModel.from_pretrained(params['transformers_model'])
params['model'] = model
debug = params['debug']
#if debug:
#    onlyfiles = onlyfiles[0:10]
    #params['epochs'] = 10
doc_id = -1
features_train = []
features_valid = []
features_test = []
t0 = time.time()
relation_stats = {0: 0, 1: 0, 2: 0, 3: 0}
for fname in tqdm.tqdm(eiid_pair_to_label.keys()):
    file_name = fname + ".tml"
    if file_name in onlyfiles_TB:
        dir_name = mypath_TB
    elif file_name in onlyfiles_AQ:
        dir_name = mypath_AQ
    elif file_name in onlyfiles_PL:
        dir_name = mypath_PL
    else:
        continue
    my_dict = tml_reader(dir_name, file_name, tokenizer)
    #TokenIDs = docTransformerTokenIDs(my_dict['sentences'])
    TokenIDs = [65]
    event_pos = []
    event_pos_end = []
    sent_end = [0]
    for sent_id, sent_dict in enumerate(my_dict['sentences']):
        last_tense = None
        subword_to_ID = sent_dict['_subword_to_ID']
        event_in_sent = {}
        for event_id in my_dict['event_dict'].keys():
            if my_dict["event_dict"][event_id]["sent_id"] == sent_id:
                event_in_sent[event_id] = {'tense': my_dict["event_dict"][event_id]['tense'],
                                           'start': my_dict['event_dict'][event_id]['_subword_id'] # 0 x x x x 0 x x x 0
                                          }
        if event_in_sent != {}:
            offset = 0
            last_event_pos = 0
            for event_id in event_in_sent:
                print("event:", my_dict["event_dict"][event_id]['mention'])
                print("original start: ", event_in_sent[event_id]['start'])
                
                start = event_in_sent[event_id]['start'] + offset
                print("orig start + offset: ", start)
                if event_in_sent[event_id]['tense'] != None:
                    if 1284 in sent_dict['_subword_to_ID'][last_event_pos+1:my_dict['event_dict'][event_id]['_subword_id']]:
                        #this_tense = event_in_sent[event_id]['tense'].replace("Past", "Future")
                        this_tense = event_in_sent[event_id]['tense'][tense_acron].replace("past", "futu")
                        #this_tense = this_tense.replace("Present", "Future")
                        this_tense = this_tense.replace("pres", "futu")
                    else:
                        this_tense = event_in_sent[event_id]['tense'][tense_acron]
                    last_tense = this_tense
                    tense_marker = tokenizer.encode(this_tense)[1:-1]
                else:
                    if last_tense != None and last_event_pos != 0:
                        print("between two events: ", tokenizer.decode(sent_dict['_subword_to_ID'][last_event_pos+1:my_dict['event_dict'][event_id]['_subword_id']]))
                        if 391 in sent_dict['_subword_to_ID'][last_event_pos+1:my_dict['event_dict'][event_id]['_subword_id']]:
                            tense_marker = tokenizer.encode(last_tense)[1:-1] # assumes that if the tense of an event not detected, it is the same as the last one
                            print("<and> appears between this event and last event, this event does not have a tense, and the tense of last event is used")
                            
                        else:
                            tense_marker = tokenizer.encode("[none]")[1:-1]
                    else:
                        tense_marker = tokenizer.encode("[none]")[1:-1]
                subword_len = len(tokenizer.encode(my_dict["event_dict"][event_id]['mention'])) - 2
                tmp_subword_to_ID = subword_to_ID[0:start] + [2589, 1736] + tense_marker + [1736] + subword_to_ID[start:start+subword_len] + [2589] + subword_to_ID[start+subword_len:]
                new_start = start + len([2589, 1736] + tense_marker + [1736]) + sent_end[-1] - 1
                print("new_start: ", new_start)
                event_pos.append(new_start)
                event_pos_end.append(new_start + subword_len)
                offset += len(tmp_subword_to_ID) - len(subword_to_ID)
                subword_to_ID = tmp_subword_to_ID
                last_event_pos = my_dict['event_dict'][event_id]['_subword_id']
        TokenIDs += subword_to_ID[1:]
        sent_end.append(len(TokenIDs))

    pairs = []
    relations = []
    
    #eiid_to_event_trigger_dict = eiid_to_event_trigger[fname]
    for (eiid1, eiid2) in eiid_pair_to_label[fname].keys():
        x = my_dict["eiid_dict"][eiid1]["eID"] # eID
        y = my_dict["eiid_dict"][eiid2]["eID"]
        pairs.append([my_dict["event_dict"][x]["event_id_why"], my_dict["event_dict"][y]["event_id_why"]])
        xy = eiid_pair_to_label[fname][(eiid1, eiid2)]
        relations.append(xy)
        relation_stats[xy] += 1
        
    feature = {'input_ids': TokenIDs,
               'event_pos': event_pos,
               'event_pos_end': event_pos_end,
               'event_pair': pairs,
               'labels': relations,
              }
    if file_name in onlyfiles_TB:
        features_train.append(feature)
    elif file_name in onlyfiles_AQ:
        features_valid.append(feature)
    elif file_name in onlyfiles_PL:
        features_test.append(feature)
elapsed = format_time(time.time() - t0)
print("MATRES Preprocessing took {:}".format(elapsed)) 
print("Temporal Relation Stats:", relation_stats)
"""
for file_name in tqdm.tqdm(onlyfiles):
    doc_id += 1
    my_dict = tsvx_reader(params['dataset'], dir_name, file_name, tokenizer, 0) # 0 if no eventseg
    TokenIDs = docTransformerTokenIDs(my_dict['sentences'])
    event_pos = []
    event_pos_end = []
    for event_id in my_dict['event_dict'].keys():
        sent_id = my_dict['event_dict'][event_id]['sent_id']
        start = my_dict['end_pos'][sent_id] - 1 + my_dict['event_dict'][event_id]['_subword_id'] # 0 x x x x 0 x x x 0
        event_pos.append(start)    
        subword_len = len(tokenizer.encode(my_dict["event_dict"][event_id]['mention'])) - 2
        event_pos_end.append(start + subword_len)
        print(tokenizer.decode([TokenIDs[start]]))
        
    pairs = []
    relations = []
    for rel in my_dict['relation_dict'].keys():
        pairs.append([rel[0], rel[1]])
        relations.append(my_dict['relation_dict'][rel]['relation'])
        
    feature = {'input_ids': TokenIDs,
               'event_pos': event_pos,
               'event_pos_end': event_pos_end,
               'event_pair': pairs,
               'labels': relations,
              }
    features.append(feature)
"""
#assert 1 == 0
#pprint(features_test[0])
#doc_num = len(onlyfiles)

if debug:
    train_dataloader = DataLoader(features_train, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_fn, drop_last=True)
    valid_dataloader = test_dataloader = train_dataloader
else:
    train_dataloader = DataLoader(features_train, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_fn, drop_last=True)
    valid_dataloader = DataLoader(features_valid, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn, drop_last=False)
    test_dataloader = DataLoader(features_test, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn, drop_last=False)
    
#raise Exception('My error!')
print("  Data processing took: {:}".format(format_time(time.time() - t0)))

OnePassModel = transformers_mlp_cons(params)
OnePassModel.to(cuda)
OnePassModel.zero_grad()
print("# of parameters:", count_parameters(OnePassModel))
for name, param in OnePassModel.named_parameters():
    if param.requires_grad:
        print(name, param.data.size())

print("batch_size:", params['batch_size'])
total_steps = len(train_dataloader) * params['epochs']
print("Total steps: [number of batches] x [number of epochs] = " + str(len(train_dataloader)) + "x" + str(params['epochs']) + " = " + str(total_steps))

mem_exp = exp(cuda, OnePassModel, params['epochs'], params['learning_rate'], train_dataloader, valid_dataloader, test_dataloader, params['dataset'], best_PATH, None, model_name)
H_F1, IC_F1 = mem_exp.train()
mem_exp.evaluate(eval_data = params['dataset'], test = True, predict = 'prediction/' + params['dataset'] + '-test-' + model_name + '.json')
# datetime object containing current date and time
now = datetime.datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
print("date and time =", dt_string)
