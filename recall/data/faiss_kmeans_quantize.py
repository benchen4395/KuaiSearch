import numpy as np
import pandas as pd
import json
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from einops import rearrange
from einops import pack
# load title embedding
data = np.load('embeddings/item_emb_512.npy')
num, dim = data.shape
print(data.shape)
batch = 16
clusters = [512,512,512]
datalen=512*4
def faiss_kmeans():
    import faiss
    global data
    codes = []
    centroids_all = []
    for k in clusters:
        kmeans = faiss.Kmeans(
            d=dim, k=k, niter=50, verbose=True,max_points_per_centroid=datalen) # 
        kmeans.train(data)
        centroids = kmeans.centroids
        _, labels = kmeans.index.search(data, 1)
        labels = labels[:, 0]
        codes.append(labels)
        data = data - centroids[labels]
        centroids_all.append(centroids)
        print(data.shape)
    codes = np.array(codes)
    print(centroids.shape)
    np.save(f'embeddings/codes_512{clusters}_{datalen}.npy', codes)



def _get_hits(query, key):
    return (rearrange(key, "b d -> 1 b d") == rearrange(query, "b d -> b 1 d")).all(axis=-1)
def code_check():
    import torch
    from einops import rearrange, pack
    from tqdm import tqdm
    codes = torch.tensor(
        np.load(f'embeddings/codes_512{clusters}_{datalen}.npy')).cuda().transpose(1,0)
    print(codes.shape)
    num = codes.shape[0]
    dedup_dim = []
    itr_codes = num // batch + 1 if num % batch else num // batch
    for i in tqdm(range(itr_codes)):
        code_query = codes[i*batch:(i+1)*batch]
        code_key = codes[:i*batch]
        is_hit = _get_hits(code_query, code_query)
        hits = torch.tril(is_hit, diagonal=-1).sum(axis=-1)
        if i:
            is_hit = _get_hits(code_query, code_key)
            hits += is_hit.sum(axis=-1)
        dedup_dim.append(hits)
    dedup_dim_array = pack(dedup_dim, "*")[0]
    corpus_ids = pack([codes, dedup_dim_array], "b *")[0]

    usage_list = []
    max_duplicates = corpus_ids[:, -1].max()
    duplicates_rate = (corpus_ids[:, -1] > 0).sum() / corpus_ids.shape[0]
    torch.save(corpus_ids, 'item_code.pt')





if __name__ == '__main__':
    faiss_kmeans()
    code_check()
