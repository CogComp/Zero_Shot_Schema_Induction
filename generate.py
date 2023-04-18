import os
from os import listdir
from os.path import isfile, join
import openai
openai.api_key = "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"

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

from timeit import default_timer as timer
import time
from datetime import timedelta
import pickle

def call_openai_api(prompt, event, n, temperature, stop, presence_penalty):
    if event:
        #prompt="The headline of the news about " + event + " was '"
        prompt="Write a news headline about " + event + ", \""
        print("--- Generating headlines for '" + event + "' ...")
    else:
        prompt="Write a news story titled \"" + prompt + "\""
        print("--- Generating text for '" + prompt + "' ...")
    print(prompt)
    response = openai.Completion.create(
        #engine="davinci",
        engine="davinci-instruct-beta-v3",
        prompt=prompt,
        max_tokens=512,
        temperature=temperature,
        stop=stop,
        n=n,
        presence_penalty=presence_penalty
    )
    texts = []
    for choice in response["choices"]:
        texts.append(choice["text"])
    print("This api call ended!")
    return texts, response["id"]

def filter_gt_sbert(generated_text, topic):
    # https://www.sbert.net/docs/usage/semantic_textual_similarity.html
    num = len(generated_text)
    topic_ = [topic] * num
    embeddings1 = model.encode(generated_text, convert_to_tensor=True)
    embeddings2 = model.encode(topic_, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    ranking = []
    for i in range(num):
        ranking.append({'index': i, 'score': cosine_scores[i][i]})
    ranking = sorted(ranking, key=lambda x: x['score'], reverse=True)
    new_gt = []
    count = -1
    for rank in ranking:
        count += 1
        if count < num / 2:
            new_gt.append(generated_text[rank['index']])
    return new_gt

def save_generated_text(generated_text):
    time_str = time.strftime("%Y-%m-%d", time.localtime(time.time()))
    with open('generated_text/' + time_str + '.pkl', 'wb') as f:
        pickle.dump(generated_text, f)
        
#scenarios = ['Bombing Attacks', 'Pandemic Outbreak', 'Civil Unrest', 'International Conflict', 'Disaster and Rescue', 'Terrorism Attacks', 'Election', 'Sports Games', 'Kidnapping', 'Business Change', 'Mass Shooting']
#scenarios = ['International Conflict', 'Mass Shooting']
dir_name = "/shared/kairos/Data/LDC2020E25_KAIROS_Schema_Learning_Corpus_Phase_1_Complex_Event_Annotation_V4/docs/ce_profile"
onlyfiles = [f for f in listdir(dir_name) if isfile(join(dir_name, f)) and f[-4:] == ".txt"]

scenarios = []
for f in onlyfiles:
    print(" ".join(f.split("_")[2:-1]))
    scenarios.append(" ".join(f.split("_")[2:-1]))

generated_text = {}
headline_n = 20
gen_n = 4

for scenario in scenarios:
    headlines, response_id = call_openai_api("", scenario.lower(), headline_n, 0.9, "\"", 0.1)
    headlines = filter_gt_sbert(headlines, scenario) # headline_n / 2
    text_gpt3 = []
    for headline in headlines:
        texts, response_id = call_openai_api(headline, None, gen_n, 0.9, None, 0.1) # headline_n / 2 * gen_n
        for text in texts:
            text_gpt3.append(headline + '. ' + text)
    """
    headline = "How to make " + scenario.lower() + " possible"
    texts, response_id = call_openai_api(headline, None, int(headline_n / 2 * gen_n), 0.9, None, 0.2) # headline_n / 2 * gen_n
    for text in texts:
        text_gpt3.append(headline + text)
    """
    generated_text[scenario] = filter_gt_sbert(text_gpt3, scenario) # headline_n / 2 * gen_n / 2
    
#print(generated_text)
save_generated_text(generated_text)
