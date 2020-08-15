# instamap: algorithm's (hyper)parameters

num_iterations = 3
k_num_dict_neighbours = 70

# embeddings and vocabularies

source_vectors = "./sample_data/vectors/unaligned/de.vectors"
# for starting from VecMap aligned vectors: "./sample_data/vectors/vecmap_aligned/de-tr.de.vectors"

source_vocabulary = "./sample_data/vectors/unaligned/de.vocab"
# for starting from VecMap aligned vectors: "./sample_data/vectors/vecmap_aligned/de-tr.de.vocab"

target_vectors = "./sample_data/vectors/unaligned/tr.vectors"
# for starting from VecMap aligned vectors: "./sample_data/vectors/vecmap_aligned/de-tr.tr.vectors"

target_vocabulary = "./sample_data/vectors/unaligned/tr.vocab"
# for starting from VecMap aligned vectors: "./sample_data/vectors/vecmap_aligned/de-tr.tr.vocab"

# translation ("training") dictionary
train_dict = "./sample_data/dicts/de-tr.train.5k.tsv"

# output (after projection)
source_vectors_output = "mapped.de.vectors"
target_vectors_output = "mapped.tr.vectors"






