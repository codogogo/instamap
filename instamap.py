import utils
import projection
import mapper
import config as c
from datetime import datetime

# loading source and target language vectors and vocabularies
vocab_src, vecs_src = utils.deserialize(c.source_vocabulary, c.source_vectors)
vocab_trg, vecs_trg = utils.deserialize(c.target_vocabulary, c.target_vectors)

# inverse vocabulary dictionaries, used when looking up mutual nearest neighbours in each iteration of InstaMap
inv_vocab_src = {v:k for k, v in vocab_src.items()}
inv_vocab_trg = {v:k for k, v in vocab_trg.items()}

# loading the translation dictionary (assumed format: one word pair per line, each line: source_language_word TAB target_language_word)
trans_dict = [x.lower().split("\t") for x in utils.load_lines(c.train_dict)]

### InstaMap
# initializing projections with original (pre-mapping) vectors
proj_src, proj_trg = vecs_src, vecs_trg

print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " Starting InstaMap mapping...")

train_d = trans_dict
# InstaMap iterations
for i in range(c.num_iterations):
  print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " Iteration: " + str(i+1))
    
  # global alignment with the Kabsch algorithm
  proj_src, proj_trg = projection.map(vocab_src, vocab_trg, proj_src, proj_trg, train_d)
  
  # local, instance-based mapping refinement
  proj_src, proj_trg = mapper.instance_map(vocab_src, proj_src, vocab_trg, proj_trg, train_d, k_closest = c.k_num_dict_neighbours)

  if i < c.num_iterations - 1:
    # identifying mutual nearest neighbours (become the tran_d for the next iteration)
    nns = utils.mutual_nn(inv_vocab_src, proj_src, inv_vocab_trg, proj_trg, train_d)
    train_d = nns
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " Size of new trans dict: " + str(len(train_d)))

print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " Serializing mapped vectors of the source language...")
utils.serialize_vectors(proj_src, c.source_vectors_output)

print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " Serializing mapped vectors of the target language...")
utils.serialize_vectors(proj_trg, c.target_vectors_output)

print(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + " I'm all done here, ciao bella!")



