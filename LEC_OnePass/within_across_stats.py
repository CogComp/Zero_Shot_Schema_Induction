import tqdm
import time
import datetime
import random
import numpy as np
from document_reader import *
from os import listdir
from os.path import isfile, join
import sys

def data(dataset, debugging, downsample, batch_size):
    train_set_HIEVE = []
    valid_set_HIEVE = []
    test_set_HIEVE = []
    train_set_MATRES = []
    valid_set_MATRES = []
    test_set_MATRES = []
    within = [0, 0, 0, 0]
    cross = [0, 0, 0, 0]
    segment_avg = 0
    if dataset in ["HiEve", "Joint"]:
        # ========================
        #       HiEve Dataset
        # ========================
        dir_name = "./hievents_v2/processed/"
        onlyfiles = [f for f in listdir(dir_name) if isfile(join(dir_name, f)) and f[-4:] == "tsvx"]
    else:
        dir_name = "./IC/IC_Processed/"
        onlyfiles = [f for f in listdir(dir_name) if isfile(join(dir_name, f)) and f[-4:] == "tsvx"]
        
    if True:
        #t0 = time.time()
        doc_id = -1
        for file_name in tqdm.tqdm(onlyfiles):
            #print(time.time() - t0)
            doc_id += 1
            if True:
                my_dict = tsvx_reader(dataset, dir_name, file_name)
                #print(file_name, my_dict['segments'])
                segment_avg += len(my_dict['segments'])
                num_event = len(my_dict["event_dict"])
                # range(a, b): [a, b)
                for x in range(1, num_event+1):
                    for y in range(x+1, num_event+1):
                        xy = my_dict["relation_dict"][(x, y)]["relation"]
                        x_seg_id = my_dict["event_dict"][x]["segment_id"]
                        y_seg_id = my_dict["event_dict"][y]["segment_id"]
                        
                        if x_seg_id==y_seg_id:
                            within[xy] += 1
                        else:
                            cross[xy] += 1

    return within, cross, segment_avg/100

w_IC, c_IC, s_IC = data("IC", 0, 0.01, 40)
w_HiEve, c_HiEve, s_HiEve = data("HiEve", 0, 0.01, 40)
print(w_IC, c_IC, s_IC)
print(w_HiEve, c_HiEve, s_HiEve)
ratio = (w_IC[0] + w_IC[1] + w_HiEve[0] + w_HiEve[1])/(w_IC[0] + c_IC[0] + w_IC[1] + c_IC[1] + w_HiEve[0] + c_HiEve[0] + w_HiEve[1] + c_HiEve[1])
print(ratio)