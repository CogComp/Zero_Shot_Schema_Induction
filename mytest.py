import requests
import json

def view_map_update(output):
    count = 0
    view_map = {}
    for view in output['views']:
        view_map[view['viewName']] = count
        count += 1
    return view_map

def event_extractor(text, text_id, NOM=True):
    if text == '':
        return {}
    headers = {'Content-type':'application/json'}
    SRL_response = requests.post('http://dickens.seas.upenn.edu:4039/annotate', json={"sentence": text}, headers=headers)
    if SRL_response.status_code != 200:
        print("SRL_response:", SRL_response.status_code)
    try:
        SRL_output = json.loads(SRL_response.text)
        return SRL_output
    except:
        print("load failed")
        return {}
    
NOM = True
SRL_output = event_extractor("The police eliminated the pro-independence army to restore order again.", 0, NOM)
SRL_view_map = view_map_update(SRL_output)
if NOM: 
    source = ['SRL_ONTONOTES', 'SRL_NOM']
else:
    source = ['SRL_ONTONOTES']
    
for viewName in source:
    print("Source:", viewName)
    for mention in SRL_output['views'][SRL_view_map[viewName]]['viewData'][0]['constituents']:
        if mention['label'] == 'Predicate':
            print(mention)