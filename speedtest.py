import pickle
with open("IE_output/Territorial Dispute_2022-01-12_21-39-46.pkl", 'rb') as f:
    IE_output = pickle.load(f)
    
from datetime import datetime 


from Information_Extractor import relation_preparer, subevent_getter, temporal_getter
temp = relation_preparer(IE_output[0])
num_event = len(temp['views'][-1]['viewData'][0]['constituents'])
num_rel = num_event * (num_event - 1) / 2
print(num_rel)
# datetime object containing current date and time
# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

subevent_res = subevent_getter(temp)
now1 = datetime.now()
print(now1 - now)

temporal_res = temporal_getter(temp)
now2 = datetime.now()
print(now2 - now1)