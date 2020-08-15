import numpy as np
from sklearn.decomposition import PCA

def pca(mat, n_comp, n_top_remove = 0):
  decomposer = PCA(n_components = n_comp) # 
  pcaed = decomposer.fit_transform(mat)
  return pcaed[:, n_top_remove:]

def build_matrices(vocab_dict_src, vocab_dict_trg, embs_src, embs_trg, trans_dict):
    src_mat = []
    trg_mat = []
    for sw, tw in trans_dict:
        if sw in vocab_dict_src and tw in vocab_dict_trg:
            src_mat.append(embs_src[vocab_dict_src[sw]])
            trg_mat.append(embs_trg[vocab_dict_trg[tw]])
    return np.array(src_mat, dtype=np.float32), np.array(trg_mat, dtype=np.float32)

def map(vocab_src, vocab_trg, embs_src, embs_trg, trans_dict, n_components = 290, n_top_remove = 0):
    print("PCA first monolingual space...") 
    pca_embs_src = pca(embs_src, n_comp = n_components, n_top_remove = n_top_remove)
    
    print("PCA second monolingual space...") 
    pca_embs_trg = pca(embs_trg, n_comp = n_components, n_top_remove = n_top_remove)
    
    proj_mat = project_kabsch(vocab_src, pca_embs_src, vocab_trg, pca_embs_trg, trans_dict)

    proj_pca_embs_src = np.matmul(pca_embs_src, proj_mat)
    return proj_pca_embs_src, pca_embs_trg


def project_kabsch(vocab_dict_src, embs_src, vocab_dict_trg, embs_trg, trans_dict):
  src_mat, trg_mat = build_matrices(vocab_dict_src, vocab_dict_trg, embs_src, embs_trg, trans_dict)
  product = np.matmul(src_mat.transpose(), trg_mat)
  U, S, V = np.linalg.svd(product)

  d = (np.linalg.det(U) * np.linalg.det(V)) < 0.0
  if(d):
    S[-1] = -S[-1]
    U[:,-1] = -U[:,-1]

  return np.matmul(U, V)
  