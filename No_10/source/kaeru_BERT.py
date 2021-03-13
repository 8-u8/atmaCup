from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import torch
import transformers

from transformers import BertTokenizer
from tqdm import tqdm
tqdm.pandas()


def result_description_pca(data, N):
    pca = PCA(n_components=N).fit(data)
    transformed = pca.transform(data)
    for n in range(N):
        print(f'第 {n+1} 主成分：{pca.explained_variance_ratio_[n]}')
    return pd.DataFrame(transformed)


def result_description_tsne(data, N):
    tsne = TSNE(n_components=N)
    transformed = tsne.fit_transform(data)
    return pd.DataFrame(transformed)


def result_description_svd(data, N):
    svd = TruncatedSVD(n_components=N).fit(data)
    transformed = svd.transform(data)
    for n in range(N):
        print(f'第 {n+1} 成分：{svd.explained_variance_ratio_[n]}')
    return pd.DataFrame(transformed)


class BertSequenceVectorizer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = "bert-base-multilingual-cased"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = transformers.BertModel.from_pretrained(
            self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.madata_len = 128

    def vectorize(self, sentence: str) -> np.array:
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)

        if len_inp >= self.madata_len:
            inputs = inp[:self.madata_len]
            masks = [1] * self.madata_len
        else:
            inputs = inp + [0] * (self.madata_len - len_inp)
            masks = [1] * len_inp + [0] * (self.madata_len - len_inp)

        inputs_tensor = torch.tensor(
            [inputs], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)

        bert_out = self.bert_model(inputs_tensor, masks_tensor)
        seq_out, pooled_out = bert_out['last_hidden_state'], bert_out['pooler_output']

        if torch.cuda.is_available():
            # 0番目は [CLS] token, 768 dim の文章特徴量
            return seq_out[0][0].cpu().detach().numpy()
        else:
            return seq_out[0][0].detach().numpy()
