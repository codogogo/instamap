# InstaMap
Instance-Based Mapping for Induction of Cross-Lingual Word Embedding Spaces

This repository accompanies the following ACL 2020 publication: 

Goran Glavaš and Ivan Vulić. Non-Linear Instance-Based Cross-Lingual Mapping for Non-Isomorphic Embedding Spaces. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL), pages 7548-7555, 2020. 

If you are using InstaMap code in your work, please cite the above paper. Here's a Bibtex entry: 
```
@inproceedings{glavas-vulic-2020-non,
    title = "Non-Linear Instance-Based Cross-Lingual Mapping for Non-Isomorphic Embedding Spaces",
    author = "Glava{\v{s}}, Goran  and
      Vuli{\'c}, Ivan",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.675",
    doi = "10.18653/v1/2020.acl-main.675",
    pages = "7548--7555"
}
```

## Code & configuration

In order create a cross-lingual embedding space with InstaMap you need to run the script *instamap.py*. The script takes no command-line parameters -- in contrast, parameter values need to be set in advance in *config.py*. The following configurable parameters exist in *config.py*:

- *num_iterations*: InstaMap is an iterative algorithm and this parameters specifies the number of iterations (empirically, setting this parameter to 3 of 4 yields best results for most language pairs)
- *k_num_dict_neighbours*: number of nearest dictionary neighbours (parameter denoted *k* in the paper) that are considered when computing "personalized" update/translation vector for each instances from the embedding space
- *source_vectors*: path to the serializted numpy matrix containing embedding vectors from the (monolingual) space of the source (first) language
- *source_vocabulary*: path to the pickled dictionary mapping words from the (monolingual) vocabulary of the first language to row indices of the embedding matrix *source_vectors*
- *target_vectors*: path to the serializted numpy matrix containing embedding vectors from the (monolingual) space of the target (second) language
- *target_vocabulary*: path to the pickled dictionary mapping words from the (monolingual) vocabulary of the second language to row indices of the embedding matrix *target_vectors*
- *train_dict*: path to the initial word translation dictionary (InstaMap is a supervised, albeit non-parametric, model/algorithm)
- *source_vectors_output*: path to which the serialize the produced mapped vectors of the source language
- *target_vectors_output*: path to which the serialize the produced mapped vectors of the target language

## Sample data

In order to quickly test InstaMap, we prepared sample data for inducing a shared space for a sample language pair, German (DE) - Turkish (TR). You can download the sample data from: 
https://drive.google.com/file/d/1iesi3gO8bAUsyuHZHNrW32MBaAk-d5zr/view?usp=sharing

In the archive *sample_data* you will find two subdirectiories: 

1. *dicts* contains the training and evaluation word translation dictionaries for this language pair (see https://github.com/codogogo/xling-eval for more detail)
2. *vectors* contains serialized vectors and vocabulary files for this two languages; in subdirectory *unaligned* you will find the pre-trained monolingual fasttext vectors of 200K most frequent terms in each language. The directory *vecmap_aligned* contains the embedding spaces of these two languages that have already been mapped into the same space with VecMap (see https://github.com/artetxem/vecmap) -- InstaMap is able to improve via instance-based mapping cross-lingual spaces induced by other methods (see the paper and IM°VM for details).   

Data for all other language pairs we experimented with in the paper can be obtained from the other repository: https://github.com/codogogo/xling-eval 

## Helper scripts

The XLing-Eval repository (https://github.com/codogogo/xling-eval) also contains some helper scripts that can help you prepare the data for InstaMap's required input formats and evaluate the quality of the induced cross-lingual spaces: 

- *code/emb_serializer.py*: Converts an embedding file given in a standard "word2vec" and "GloVe" textual format into a pair of vector matrix (.vectors file) and vocabulary dictionary (.vocab file) files that you need as input for InstaMap
- *code/eval.py*: once the shared embedding space has been induced with InstaMap, you can evaluate its BLI performance using the script *eval.py* from XLing-Eval repo.
- *code/emb_deserializer.py*: converts the serialized "matrix and vocabulary" files (.vectors, .vocab) for a language back into a single textual word embeddings file

