import numpy as np
sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']



# Embeedding模型GanymedeNil/text2vec-large-chinese是一个setence embedding model，我们知道word embedding是把word或sub-word编码成一定维度的词向量的过程.
# 这里的setence embedding model就是将输入的一个句子编码成句向量的过程，核心原理都是相通的。
# text2vec-large-chinese
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
em1 = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese", model_kwargs={'device': "cuda"})
sentence_embeddings1 = em1.embed_documents(sentences)
sentence_embeddings1 = np.array(sentence_embeddings1)
print("sentence_embeddings1 shape=", sentence_embeddings1.shape)
vec1, vec2 = sentence_embeddings1
cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print("余弦相似度：%.3f"%cos_sim)





# 原始的
# import numpy as np
# sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']
#
# # text2vec-base-chinese
# from sentence_transformers import SentenceTransformer
# m = SentenceTransformer("shibing624/text2vec-base-chinese")
# sentence_embeddings = m.encode(sentences)
# print("sentence_embeddings shape=", sentence_embeddings.shape)
# vec1, vec2 = sentence_embeddings
# cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
# print("余弦相似度：%.3f"%cos_sim)
