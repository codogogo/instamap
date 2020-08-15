import numpy as np
import pickle
import codecs

# IO

def load_lines(path):
	return [l.strip() for l in list(codecs.open(path, "r", encoding = 'utf8', errors = 'replace').readlines())]

def deserialize(path_vocab, path_vecs): 
    vectors = np.load(path_vecs)
    vocab = pickle.load(open(path_vocab,"rb"))
    return vocab, vectors 

def serialize_vectors(vectors, path):
  np.save(open(path,"wb+"), vectors)
  
# Matrix / Linear algebra operations

def mat_normalize(mat, norm_order=2, axis=1):
  return mat / np.transpose([np.linalg.norm(mat, norm_order, axis)])

def mat_mean_center(mat):
  col_means = np.mean(mat, axis = 0)
  return mat - col_means

# mutual NNs

def big_matrix_multiplication(a, b, function_on_result, chunk_size = 100):
  result = []
  num_iters = a.shape[0] // chunk_size + (0 if a.shape[0] % chunk_size == 0 else 1)
  for i in range(num_iters):
    print("Batch multiplication iter: " + str(i+1))
    mul_batch = np.dot(a[i * chunk_size : (i+1) * chunk_size, :], b)
    res_batch = function_on_result(mul_batch)
    result.extend(res_batch)
  return np.array(result)

def mutual_nn(inv_vocab_src, proj_src, inv_vocab_trg, proj_trg, train_dict):
  src = mat_normalize(proj_src)  
  trg = mat_normalize(proj_trg)

  ind_src_trg = big_matrix_multiplication(src, np.transpose(trg), lambda x: np.argmax(x, axis = 1), chunk_size = 30000)
  ind_trg_src = big_matrix_multiplication(trg, np.transpose(src), lambda x: np.argmax(x, axis = 1), chunk_size = 30000)

  nns = [(inv_vocab_src[i], inv_vocab_trg[ind_src_trg[i]]) for i in range(len(ind_src_trg)) if ind_trg_src[ind_src_trg[i]] == i]
  print("Number of NN matches: " + str(len(nns)))
  return nns