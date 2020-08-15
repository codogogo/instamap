import numpy as np
import projection
import utils
from datetime import datetime

def instance_map(vocab_dict_src, embs_src, vocab_dict_trg, embs_trg, trans_dict, k_closest = 10):
    src_mat, trg_mat = projection.build_matrices(vocab_dict_src, vocab_dict_trg, embs_src, embs_trg, trans_dict)
    diff_mat = (trg_mat - src_mat)

    ## Mapping/tuning of source space vectors
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " Computing dictionary closest neighbours for all source space vectors...")
    embs_src_norm = utils.mat_normalize(embs_src)  
    src_mat_norm = utils.mat_normalize(src_mat)

    cosines = np.matmul(embs_src_norm, np.transpose(src_mat_norm))
    closest = np.flip(np.argsort(cosines, axis = 1), axis = 1)[:, :k_closest]
    
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " Tuning/mapping source space vectors...")
    proj_src = []
    for i in range(embs_src.shape[0]):
        indices = closest[i]
        cosines_closest = cosines[i][closest[i]]
        cosines_closest = cosines_closest / np.sum(cosines_closest)
    
        dir_mat = diff_mat[indices, :]
        cos_weights = np.tile(np.reshape(cosines_closest, (len(cosines_closest), 1)), (1, dir_mat.shape[1]))
        
        dir_vec = np.sum(np.multiply(dir_mat, cos_weights), axis = 0)     
        y_pr = embs_src[i] + dir_vec
        
        proj_src.append(y_pr)

    ## Mapping/tuning of source space vectors
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " Computing dictionary closest neighbours for all target space vectors...")
    embs_trg_norm = utils.mat_normalize(embs_trg)  
    trg_mat_norm = utils.mat_normalize(trg_mat)
    
    cosines = np.matmul(embs_trg_norm, np.transpose(trg_mat_norm))
    closest = np.flip(np.argsort(cosines, axis = 1), axis = 1)[:, :k_closest]
    
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " Tuning/mapping target space vectors...")
    proj_trg = [] 
    for i in range(embs_trg.shape[0]):
        indices = closest[i]
        cosines_closest = cosines[i][closest[i]]
        cosines_closest = cosines_closest / np.sum(cosines_closest)
    
        dir_mat = diff_mat[indices, :]
        cos_weights = np.tile(np.reshape(cosines_closest, (len(cosines_closest), 1)), (1, dir_mat.shape[1]))
        
        dir_vec = np.sum(np.multiply(dir_mat, cos_weights), axis = 0)     
        y_pr = embs_trg[i] - dir_vec
        
        proj_trg.append(y_pr)

    return np.array(proj_src), np.array(proj_trg)