from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def filter_gt_sbert(generated_text, topic):
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

print(filter_gt_sbert(['The cat is perfect.', 'Columbia got bombing threat this morning.', 'Black lives matter.', 'I love you'], ['Bombing attack'] * 4))
# output: 
# ['Columbia got bombing threat this morning.', 'I love you']
