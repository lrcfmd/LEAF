#!/usr/bin/env python
# coding: utf-8

# # Element Similarity
# 
# This notebook is used to reproduce the plots shown in the paper.

# In[ ]:


# Imports
import sys
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from elementembeddings.core import Embedding, data_directory
from elementembeddings.plotter import dimension_plotter, heatmap_plotter
import pandas as pd
import os
import seaborn as sns

sns.set_context("paper", font_scale=1.5)
random_state = 42


# ## Introduction
# 
# Let's set up the Embedding classes and load the data

# In[ ]:


# Load the embeddings
cbfvs = [
        "leaf",
    "magpie",
    "matscholar",
    "mat2vec",
    "megnet16",
    "oliynyk",
    "random_200",
    "skipatom",
]
element_embedddings = {cbfv: Embedding.load_data(cbfv) for cbfv in cbfvs}

print('Loaded embeddings')
# We can reproduce some of the information in table I from the paper by running the following code:

# Let's find the dimensionality of all of the CBFVs that we have loaded

element_embedddings_dim = {cbfv: [element_embedddings[cbfv].dim] for cbfv in cbfvs}

dim_df = pd.DataFrame.from_dict(
    element_embedddings_dim, orient="index", columns=["dimension"]
)
print(dim_df)


# ## II.B Similarity measures

# Let's set up the Embedding classes for our analysis

# Standardise the representations
for embedding in element_embedddings.values():
    print(f"Attempting to standardise {embedding.embedding_name}...")
    print(f" Already standardised: {embedding.is_standardised}")
    embedding.standardise(inplace=True)
    print(f"Now standardised: {embedding.is_standardised}")

# Get our four embeddings to compare
cbfvs_to_keep = ["matscholar", "mat2vec", "megnet16", "leaf"]
element_vectors = {cbfv: element_embedddings[cbfv] for cbfv in cbfvs_to_keep}

# Keep the first 83 elements

# Get the ordered symbols file
symbols_path = os.path.join(data_directory, "element_data", "ordered_periodic.txt")
with open(symbols_path) as f:
    symbols = f.read().splitlines()

# Get the first 83 elements
symbols = symbols[:83]

for cbfv in cbfvs_to_keep:
    # Get the keys of the atomic embeddings object
    elements = set(element_vectors[cbfv].element_list)
    el_symbols_set = set(symbols)

    # Get the element symbols we want to remove
    els_to_remove = list(elements - el_symbols_set)

    # Iteratively delete the elements with atomic number
    # greater than 83 from our embeddings
    for el in els_to_remove:
        del element_vectors[cbfv].embeddings[el]

    # Verify that we have 83 elements
    print(len(element_vectors[cbfv].element_list))


# ### Distances and similarities

distances = ["euclidean", "manhattan", "chebyshev"]
for distance in distances:
    d = element_embedddings["magpie"].compute_distance_metric("Li", "K", distance)
    d_Li_Bi = element_embedddings["magpie"].compute_distance_metric(
        "Li", "Bi", distance
    )
    print(f"Distance between Li and K using {distance} is {d:.2f}")
    print(f"Distance between Li and Bi using {distance} is {d_Li_Bi:.2f}")

# Get the pearson correlation and cosine similarity between the embeddings for Li and K
similarity_metrics = ["pearson", "cosine_similarity"]
for similarity_metric in similarity_metrics:
    magpie_d = element_embedddings["magpie"].compute_correlation_metric(
        "Li", "K", similarity_metric
    )

    magpie_d_Li_Bi = element_embedddings["magpie"].compute_correlation_metric(
        "Li", "Bi", similarity_metric
    )

    mvec_d = element_embedddings["mat2vec"].compute_correlation_metric(
        "Li", "K", similarity_metric
    )
    mvec_d_Li_Bi = element_embedddings["mat2vec"].compute_correlation_metric(
        "Li", "Bi", similarity_metric
    )

    print(
        f"The metric, {similarity_metric}, between Li and K is {magpie_d:.3f} for magpie and {mvec_d:.3f} for mat2vec"
    )
    print(
        f"The metric, {similarity_metric}, between Li and Bi is {magpie_d_Li_Bi:.3f} for magpie and {mvec_d_Li_Bi:.3f} for mat2vec"
    )


# ### Euclidean distances
# 
# 
# \begin{equation}
# d_E(\textbf{A,B}) = 
# \sqrt{
# (A_1 - B_1)^2 
# + \cdots
# + (A_n - B_n)^2 }
# \end{equation}
# 
# We can use the Euclidean distance to compare the similarity of two elements. The following code will plot the distribution of the Euclidean distances between all pairs of elements in the embedding space.

# In[ ]:


fig, (axes) = plt.subplots(2, 2, figsize=(10, 10))

for ax, cbfv in zip(axes.flatten(), cbfvs_to_keep):
    heatmap_plotter(
        embedding=element_vectors[cbfv],
        metric="euclidean",
        sortaxisby="atomic_number",
        show_axislabels=False,
        ax=ax,
    )

plt.tight_layout()
plt.savefig("1_euclidean.pdf")
plt.show()


# ### Manhattan distances
# 
# \begin{equation}
# d_M(\textbf{A,B}) = 
# \sum_{i=1}^n |A_i - B_i|
# \end{equation}
# 
# We can use the Manhattan distance to compare the similarity of two elements. The following code will plot the distribution of the Manhattan distances between all pairs of elements in the embedding space.
# 

# In[ ]:


fig, (axes) = plt.subplots(2, 2, figsize=(10, 10))

for ax, cbfv in zip(axes.flatten(), cbfvs_to_keep):
    heatmap_plotter(
        embedding=element_vectors[cbfv],
        metric="manhattan",
        sortaxisby="atomic_number",
        show_axislabels=False,
        ax=ax,
    )

plt.tight_layout()
plt.savefig("2_manhattan.pdf")
plt.show()


# ### Cosine similarity
# 
# \begin{equation}
# cos(\theta) = \frac{\textbf{A} \cdot \textbf{B}} {||\textbf{A}|| ||\textbf{B}||}
# \end{equation}

# In[ ]:


fig, (axes) = plt.subplots(2, 2, figsize=(10, 10))
heatmap_params = {"vmin": -1, "vmax": 1}
for ax, cbfv in zip(axes.flatten(), cbfvs_to_keep):
    heatmap_plotter(
        embedding=element_vectors[cbfv],
        metric="cosine_similarity",
        sortaxisby="atomic_number",
        show_axislabels=False,
        cmap="Blues_r",
        ax=ax,
        **heatmap_params
    )

#p = embedding.correlation_pivot_table(metric="cosine_similarity", sortby="atomic_number"  )
#print(p.head())
#p.to_pickle('leaf200_cosine_similarity.pickle')

plt.tight_layout()
#plt.savefig("3_cosine_similarity.pdf")
plt.show()

# ### Pearson correlation
# 
# \begin{equation}
# \rho_{A,B} = \frac{cov(A,B)}{\sigma_{A}\sigma_{B}}
# \end{equation}

# In[ ]:


fig, (axes) = plt.subplots(2, 2, figsize=(10, 10))
heatmap_params = {"vmin": -1, "vmax": 1}
for ax, cbfv in zip(axes.flatten(), cbfvs_to_keep):
    heatmap_plotter(
        embedding=element_vectors[cbfv],
        metric="pearson",
        sortaxisby="atomic_number",
        show_axislabels=False,
        cmap="Blues_r",
        ax=ax,
        **heatmap_params
    )

plt.tight_layout()
plt.savefig("4_pearson.pdf")
plt.show()

sys.exit(0)
# ## II.C Dimensionality reduction
# To visualise the embeddings, we can use dimensionality reduction techniques such as PCA and t-SNE. The following code will plot the embeddings in 2D using PCA, t-SNE and UMAP.

# ### Principal Component Analysis (PCA)
# 
# The main concept behind PCA is to reduce the dimensionality of a dataset consisting of many variables correlated with each other, either heavily or lightly, while retaining the variation present in the dataset, up to the maximum extent.

# In[ ]:


fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 10),
)
reducer_params = {"random_state": random_state}
scatter_params = {"s": 80}
for ax, cbfv in zip(axes.flatten(), cbfvs_to_keep):
    dimension_plotter(
        embedding=element_vectors[cbfv],
        reducer="pca",
        n_components=2,
        ax=ax,
        adjusttext=True,
        reducer_params=reducer_params,
        scatter_params=scatter_params,
    )
    ax.legend().remove()

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.54, 1.06), loc="upper center", ncol=5)
fig.tight_layout()
plt.savefig("5_pca.pdf", bbox_inches="tight")
fig.show()


# ### t-Distributed Stochastic Neighbor Embedding (t-SNE)
# 
# t-SNE is a non-linear dimensionality reduction technique that is particularly well-suited for embedding high-dimensional data into a space of two or three dimensions, which can then be visualized in a scatter plot.

# In[ ]:


fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 10),
)
scatter_params = {"s": 80}
reducer_params = {"random_state": random_state}
for ax, cbfv in zip(axes.flatten(), cbfvs_to_keep):
    dimension_plotter(
        embedding=element_vectors[cbfv],
        reducer="tsne",
        n_components=2,
        ax=ax,
        adjusttext=True,
        scatter_params=scatter_params,
        reducer_params=reducer_params,
    )
    ax.legend().remove()

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.54, 1.06), loc="upper center", ncol=5)
fig.tight_layout()
plt.savefig("6_tsne.pdf", bbox_inches="tight")
fig.show()


# ### Uniform Manifold Approximation and Projection (UMAP)
# 
# UMAP is a dimension reduction technique that can be used for visualisation similarly to t-SNE, but also for general non-linear dimension reduction. The algorithm is founded on three assumptions about the data: the data is uniformly distributed on a Riemannian manifold, the Riemannian metric is locally constant, and the manifold is locally connected. UMAP is constructed from a theoretical framework based in Riemannian geometry and algebraic topology. The result is a practical scalable algorithm that applies to real world data.

# In[ ]:


fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 10),
)

reducer_params = {"random_state": random_state}
scatter_params = {"s": 80}

for ax, cbfv in zip(axes.flatten(), cbfvs_to_keep):
    dimension_plotter(
        embedding=element_vectors[cbfv],
        reducer="umap",
        n_components=2,
        ax=ax,
        adjusttext=True,
        reducer_params=reducer_params,
        scatter_params=scatter_params,
    )
    ax.legend().remove()

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.54, 1.06), loc="upper center", ncol=5)
fig.tight_layout()
plt.savefig("7_umap.pdf", bbox_inches="tight")
fig.show()

