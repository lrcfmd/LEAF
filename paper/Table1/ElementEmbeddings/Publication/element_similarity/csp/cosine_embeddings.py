#!/usr/bin/env python
# coding: utf-8

# # Generating pairwise cosine similarities

# In[ ]:


import json
import pandas as pd
import itertools
from elementembeddings.core import Embedding, data_directory
from smact.structure_prediction.utilities import parse_spec
import smact
from smact.data_loader import (
    lookup_element_oxidation_states_icsd,
    lookup_element_oxidation_states,
    lookup_element_oxidation_states_sp,
)
from smact.structure_prediction.utilities import unparse_spec
import os

# Load the embeddings
cbfvs = [
        "atom2vec"
#  "leaf_neighbour",
#  "leaf2_200",
#  "leaf59",
#  "leaf200",
#    "magpie_mat2vec"
#   "magpie",
#   "matscholar",
#   "mat2vec",
#   "megnet16",
#   "oliynyk",
#   "random_200",
#   "skipatom",
]
element_embeddings = {cbfv: Embedding.load_data(cbfv) for cbfv in cbfvs}

# Standardise
for embedding in element_embeddings.values():
    print(f"Attempting to standardise {embedding.embedding_name}...")
    print(f" Already standardised: {embedding.is_standardised}")
    embedding.standardise(inplace=True)
    print(f"Now standardised: {embedding.is_standardised}")
# Keep the first 83 elements

# Get the ordered symbols file
symbols_path = os.path.join(data_directory, "element_data", "ordered_periodic.txt")
with open(symbols_path) as f:
    symbols = f.read().splitlines()

# Get the first 83 elements
symbols = symbols[:83]

for cbfv in cbfvs:
    # Get the keys of the atomic embeddings object
    elements = set(element_embeddings[cbfv].element_list)
    el_symbols_set = set(symbols)

    # Get the element symbols we want to remove
    els_to_remove = list(elements - el_symbols_set)

    # Iteratively delete the elements with atomic number
    # greater than 83 from our embeddings
    for el in els_to_remove:
        del element_embeddings[cbfv].embeddings[el]

    # Verify that we have 83 elements
    print(len(element_embeddings[cbfv].element_list))


# ### Generating list of species
# 
# The current version of the SMACT structure prediction is based on using species rather than elements. To circumvent that, we create table of pairwise lambda values for the species by assuming all species for a given element will take the same cosine similarity.
# 
# We acknowledge that this is not an accurate assumption, but in this work, with a focus on unary substitutions where we are not trying to assign structures to hypothetical compositions as well as charge-neutrality being enforced in the substitutions, this should not lead to odd predictions.

# In[ ]:


# SMACT is used to get a list of species
species = []
for element in symbols:
    oxidation_states = lookup_element_oxidation_states(element)
    for oxidation_state in oxidation_states:
        species.append(unparse_spec((element, oxidation_state)))

print(len(species))


# In[ ]:


# Generate the pairwise cosine similarities.


if not os.path.exists("cosine_similarity/"):
    os.mkdir("cosine_similarity/")
table_dict = {}
species_pairs = list(itertools.combinations_with_replacement(species, 2))
for cbfv in element_embeddings.keys():
    print(cbfv)
    table = []
    for spec1, spec2 in species_pairs:
        corr = element_embeddings[cbfv].compute_correlation_metric(
            parse_spec(spec1)[0], parse_spec(spec2)[0], metric="pearson"
        )
        table.append([spec1, spec2, corr])
        if spec1 != spec2:
            table.append([spec2, spec1, corr])
    table_dict[cbfv] = table
    with open(f"cosine_similarity/{cbfv}.json", "w") as f:
        json.dump(table, f)


# In[ ]:

