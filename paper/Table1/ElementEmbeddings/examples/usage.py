#!/usr/bin/env python
# coding: utf-8

# # Using the ElementEmbeddings package
# This notebook will serve as a tutorial for using the ElementEmbeddings package and going over the core features.

# In[1]:


# Imports
import sys
import numpy as np
import pandas as pd
import seaborn as sns

from elementembeddings.core import Embedding
from elementembeddings.plotter import heatmap_plotter, dimension_plotter
import matplotlib.pyplot as plt

sns.set(font_scale=1.5)

# ## Elemental representations
# 
# A key problem in supervised machine learning problems is determining the featurisation/representation scheme for a material in order to pass it through a mathematical algorithm. For composition only machine learning, we want to be able create a numerical representation of a chemical formula A<sub>w</sub>B<sub>x</sub>C<sub>y</sub>D<sub>z</sub>. We can achieve this by creating a composition based feature vector derived from the elemental properties of the constituent atoms or a representation can be learned during the supervised training process.
# 
# A few of these CBFV have been included in the package and we can load them using the `load_data` class method.
# 

# In[ ]:


# Create a list of the available CBFVs included in the package

cbfvs = [
        "leaf",
    "magpie",
    "mat2vec",
    "matscholar",
    "megnet16",
    "oliynyk",
    "random_200",
    "skipatom",
    "mod_petti",
    "magpie_sc",
    "oliynyk_sc",
]

# Create a dictionary of {cbfv name : Atomic_Embeddings objects} key, value pairs
AtomEmbeds = {cbfv: Embedding.load_data(cbfv) for cbfv in cbfvs}

print('Loaded embeddings!')
# Taking the magpie representation as our example, we will demonstrate some features of the the `Embedding` class.

# Let's use magpie as our example

# Let's look at the CBFV of hydrogen for the magpie representation
print(
    "Below is the CBFV/representation of the hydrogen atom from the magpie data we have \n"
)
print(AtomEmbeds["magpie"].embeddings["H"])


# We can check the elements which have a feature vector for a particular embedding


# We can also check to see what elements have a CBFV for our chosen representation
print("Magpie has composition-based feature vectors for the following elements: \n")
print(AtomEmbeds["magpie"].element_list)


# For the elemental representations distributed with the package, we also included BibTex citations of the original papers were these representations are derived from. This is accessible through the `.citation()` method.


# Print the bibtex citation for the magpie embedding
print(AtomEmbeds["magpie"].citation())


# We can also check the dimensionality of the elemental representation.

# We can quickly check the dimensionality of this CBFV
magpie_dim = AtomEmbeds["magpie"].dim
print(f"The magpie CBFV has a dimensionality of {magpie_dim}")


# Let's find the dimensionality of all of the CBFVs that we have loaded


AtomEmbeds_dim = {
    cbfv: {"dim": AtomEmbeds[cbfv].dim, "type": AtomEmbeds[cbfv].embedding_type}
    for cbfv in cbfvs
}

dim_df = pd.DataFrame.from_dict(AtomEmbeds_dim)
dim_df.T


# We can see a wide range of dimensions of the composition-based feature vectors.
# 
# Let's know explore more of the core features of the package.
# The numerical representation of the elements enables us to quantify the differences between atoms. With these embedding features, we can explore how similar to atoms are by using a 'distance' metric. Atoms with distances close to zero are 'similar', whereas elements which have a large distance between them should in theory be dissimilar. 
# 
# Using the class method `compute_distance_metric`, we can compute these distances.

# Let's continue using our magpie cbfv
# The package contains some default distance metrics: euclidean, manhattan, chebyshev

metrics = ["euclidean", "manhattan", "chebyshev", "wasserstein", "energy"]

distances = [
    AtomEmbeds["magpie"].compute_distance_metric("Li", "K", metric=metric)
    for metric in metrics
]
print("For the magpie representation:")
for i, distance in enumerate(distances):
    print(
        f"Using the metric {metrics[i]}, the distance between Li and K is {distance:.2f}"
    )


# Let's continue using our magpie cbfv
# The package contains some default distance metrics: euclidean, manhattan, chebyshev

metrics = ["euclidean", "manhattan", "chebyshev", "wasserstein", "energy"]

distances = [
    AtomEmbeds["leaf"].compute_distance_metric("Li", "K", metric=metric)
    for metric in metrics
]
print("For the leaf representation:")
for i, distance in enumerate(distances):
    print(
        f"Using the metric {metrics[i]}, the distance between Li and K is {distance:.2f}"
    )

sys.exit(0)
# ## Plotting
# We can also explore the correlation between embedding vectors.
# In the example below, we will plot a heatmap of the pearson correlation of our magpie CBFV, a scaled magpie CBFV and the 16-dim megnet embeddings

# ### Pearson Correlation plots

# #### Unscaled and scaled Magpie


fig, ax = plt.subplots(figsize=(24, 24))
heatmap_plotter(
    embedding=AtomEmbeds["magpie"],
    metric="pearson",
    sortaxisby="atomic_number",
    # show_axislabels=False,
    ax=ax,
)

fig.show()


fig, ax = plt.subplots(figsize=(24, 24))
heatmap_plotter(
    embedding=AtomEmbeds["megnet16"],
    metric="pearson",
    sortaxisby="atomic_number",
    # show_axislabels=False,
    ax=ax,
)

fig.show()

# ### PCA plots
fig, ax = plt.subplots(figsize=(16, 12))

dimension_plotter(
    embedding=AtomEmbeds["magpie"],
    reducer="pca",
    n_components=2,
    ax=ax,
    adjusttext=True,
)

fig.tight_layout()
fig.show()



fig, ax = plt.subplots(figsize=(16, 12))

dimension_plotter(
    embedding=AtomEmbeds["magpie_sc"],
    reducer="pca",
    n_components=2,
    ax=ax,
    adjusttext=True,
)

fig.tight_layout()
fig.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 12))

dimension_plotter(
    embedding=AtomEmbeds["megnet16"],
    reducer="pca",
    n_components=2,
    ax=ax,
    adjusttext=True,
)

fig.tight_layout()
fig.show()


# ### t-SNE plots

# In[ ]:


fig, ax = plt.subplots(figsize=(16, 12))

dimension_plotter(
    embedding=AtomEmbeds["magpie"],
    reducer="tsne",
    n_components=2,
    ax=ax,
    adjusttext=True,
)

fig.tight_layout()
fig.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 12))

dimension_plotter(
    embedding=AtomEmbeds["mat2vec"],
    reducer="tsne",
    n_components=2,
    ax=ax,
    adjusttext=True,
)

fig.tight_layout()
fig.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(16, 12))

dimension_plotter(
    embedding=AtomEmbeds["megnet16"],
    reducer="tsne",
    n_components=2,
    ax=ax,
    adjusttext=True,
)

fig.tight_layout()
fig.show()


# In[ ]:




