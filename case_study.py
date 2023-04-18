#import Information_Extractor
#import importlib
#importlib.reload(Information_Extractor)
from Information_Extractor import temporal_getter, relation_preparer, event_extractor, print_stats
text = input("Please enter the text:\n")
while text!='':
    #text = "The police eliminated the pro-independence army before restoring order."
    onepass = input("Use onepass: 1; Don't use onepass: 0\n")
    SRL_output = event_extractor(text, 0)
    temp = relation_preparer(SRL_output)
    temporal_res = temporal_getter(temp, onepass = int(onepass))
    with open("temp_res.txt", 'w') as f:
        print_stats([temporal_res], None, f)
    with open("temp_res.txt") as f:
        print("\n" + text)
        lines = f.readlines()
        print("\nbefore relations:\n")
        for line in lines:
            if line.startswith("\'"):
                print(line.strip())
    text = input("\nPlease enter the text:\n")