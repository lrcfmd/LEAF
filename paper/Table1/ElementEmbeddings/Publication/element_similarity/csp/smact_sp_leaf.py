#!/usr/bin/env python
# coding: utf-8

# # SMACT - Structure prediction
# 
# This notebook is used to perform the structure predictions in the publication. It uses SMACT to carry out structure substitutions.
# Imports
import sys
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
from sklearn.metrics import matthews_corrcoef as mcc
import seaborn as sns
import matplotlib.pyplot as plt
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.ext.matproj import MPRester
from typing import List
from operator import itemgetter
from datetime import datetime
from pymatgen.transformations.standard_transformations import (
    OxidationStateDecorationTransformation,
)
from monty.serialization import loadfn
from smact.structure_prediction import (
    prediction,
    database,
    mutation,
    probability_models,
    structure,
    utilities,
)
from pymatgen.core.structure import Structure as pmg_structure
import pandas as pd
import os
import re


def parse_species(species: str) -> tuple[str, int]:
    """
    Parses a species string into its atomic symbol and oxidation state.

    :param species: the species string
    :return: a tuple of the atomic symbol and oxidation state

    """

    ele = re.match(r"[A-Za-z]+", species).group(0)

    charge_match = re.search(r"\d+", species)
    ox_state = int(charge_match.group(0)) if charge_match else 0

    if "-" in species:
        ox_state *= -1

    # Handle cases of X+ or X- (instead of X1+ or X1-)
    if "+" in species and ox_state == 0:
        ox_state = 1

    if ox_state == 0 and "-" in species:
        ox_state = -1
    return ele, ox_state


# ## Load the data
print('Here we load the data from our radius ratio rules experiment.')


df = loadfn("df_final_structure.json")

df["structure"] = df["structure"].apply(lambda x: pmg_structure.from_str(x, fmt="json"))

print(df.shape)
df.head()


# The following block serves to add oxidation states to the structures so that we can add them into the smact StructureDB object.

print('Add oxidation states to the structures')

oxi_structures = []
mpid_to_comp = {}
comp_to_mpid = {}
for i, row in df.iterrows():
    spec_dict = {row["cation"][0]: row["cation"][1], row["anion"][0]: row["anion"][1]}
    oxi_trans = OxidationStateDecorationTransformation(spec_dict)
    oxi_structures.append(oxi_trans.apply_transformation(row["structure"]))

    # Create a dictionary to map between material id and formula
    mpid_to_comp[row["material_id"]] = row["formula_pretty"]
    comp_to_mpid[row["formula_pretty"]] = row["material_id"]

df["oxi_structure"] = oxi_structures
df.head()


print('Create database')
db_test = database.StructureDB("binary_structures.db")

# Uncomment the block below the first time you run this cell to create a table and add the structures.

#db_test.add_table("structures")

# Create smactstructures

smact_structs = [structure.SmactStructure.from_py_struct(struct,determine_oxi="predecorated") for struct in df['oxi_structure']]

# Add structures to database
db_test.add_structs(smact_structs,table="structures")

# ### Set up Cation Mutators
# 
cosine_cbfv_files = os.listdir("cosine_similarity")
element_embeddings = [f.split(".")[0] for f in cosine_cbfv_files]

print('CATION MUTATOR', element_embeddings)

# change:
element_embeddings = ['leaf+', 'leaf_neighbour']

CM_dict = {}
for element_embedding in element_embeddings:
    CM_dict[element_embedding] = mutation.CationMutator.from_json(
        f"cosine_similarity_leaf/{element_embedding}.json"
    )

CM_dict["hautier"] = mutation.CationMutator.from_json()

# ### Set up structure prediction functions
# 

def predict_structures(
    predictor: prediction.StructurePredictor,
    species: list[tuple[str, int]],
    thresh: float = 0,
):
    """
    Predict structures for a given species.
    """
    preds = []
    parents_list = []
    probs_list = []

    for specs in species:
        try:
            predictions = list(
                predictor.predict_structs(specs, thresh=thresh, include_same=False)
            )
            predictions.sort(key=itemgetter(1), reverse=True)
            parents = [x[2].composition() for x in predictions]
            probs = [x[1] for x in predictions]
            preds.append(predictions)
            parents_list.append(parents)
            probs_list.append(probs)
        except ValueError:
            preds.append([])
            parents_list.append([])
            probs_list.append([])

    pred_structs = []
    parent_comp = []
    for pred in preds:
        if len(pred) == 0:
            pred_structs.append(None)
            parent_comp.append(None)
        else:
            pred_structs.append(pred[0][0].as_poscar())
            parent_comp.append(
                pmg_structure.from_str(
                    pred[0][2].as_poscar(), fmt="poscar"
                ).composition.reduced_formula
            )

    print(
        len(pred_structs), len(species), len(preds), len(probs_list), len(parents_list)
    )
    return pred_structs, parent_comp


#API_KEY = os.environ["8E8EtFRXXZiRAKD4Luvm7bvsUlcdm5ax"]
API_KEY = "8E8EtFRXXZiRAKD4Luvm7bvsUlcdm5ax"
# Get the structures of the 4 chosen structure types.

struct_files = ["cscl.cif", "rock_salt.cif", "zinc_blende.cif", "wurtzite.cif"]

# Load the structures from cif files if available else, query materials project
if all([True for x in struct_files if x in os.listdir("./structure_files")]):
    cscl_struct = pmg_structure.from_file("./structure_files/cscl.cif")
    rock_salt_struct = pmg_structure.from_file("./structure_files/rock_salt.cif")
    zinc_blende_struct = pmg_structure.from_file("./structure_files/zinc_blende.cif")
    wurtzite_struct = pmg_structure.from_file("./structure_files/wurtzite.cif")
else:
    with MPRester(API_KEY) as mpr:
        cscl_struct = mpr.get_structure_by_material_id("mp-22865")
        rock_salt_struct = mpr.get_structure_by_material_id("mp-22862")
        zinc_blende_struct = mpr.get_structure_by_material_id("mp-10695")
        wurtzite_struct = mpr.get_structure_by_material_id("mp-560588")
        # Save structures to cifs
        cscl_struct.to(filename="cscl.cif")
        rock_salt_struct.to(filename="rock_salt.cif")
        zinc_blende_struct.to(filename="zinc_blende.cif")
        wurtzite_struct.to(filename="wurtzite.cif")


# Get the list of species
species_list = [
    (parse_species(x[0]), parse_species(x[1])) for x in df["possible_species"]
]

# ### Structure prediction

for key, value in CM_dict.items():
    print('STRUCTURE prediction')
    print(key, value)

    # Predict structures
    # Set up the predictor
    sp_test = prediction.StructurePredictor(
        mutator=value, struct_db=db_test, table="structures"
    )
    prediction_result = predict_structures(sp_test, species_list, thresh=0)
    df[f"{key}_struct"], df[f"{key}_formula"] = (
        prediction_result[0],
        prediction_result[1],
    )

df.head()
print(df.columns)
df.to_pickle('SMACT_PREDICTED_DB.pickle')


# Determine the structure type of the predicted structures


SM = StructureMatcher(attempt_supercell=True)


def determine_structure_type(structure):
    if structure is None:
        return None
    elif SM.fit_anonymous(pmg_structure.from_str(structure, fmt="poscar"), cscl_struct):
        return "cscl"
    elif SM.fit_anonymous(
        pmg_structure.from_str(structure, fmt="poscar"), rock_salt_struct
    ):
        return "rock salt"
    elif SM.fit_anonymous(
        pmg_structure.from_str(structure, fmt="poscar"), zinc_blende_struct
    ):
        return "zinc blende"
    elif SM.fit_anonymous(
        pmg_structure.from_str(structure, fmt="poscar"), wurtzite_struct
    ):
        return "wurtzite"
    else:
        return "other"

for key in CM_dict.keys():
    df[f"{key}_struct_type"] = df[f"{key}_struct"].apply(determine_structure_type)

df.head()

df.to_pickle('smact_predicted_AB.pickle')

# Determine if the predicted structure type is the same as the original structure type

for key in CM_dict.keys():
    df[f"{key}_same_struct_type"] = df.apply(
        lambda x: x["structure_type"] == x[f"{key}_struct_type"], axis=1
    )

df.head()

# Create a dictionary of value counts for each structure type
struct_type_counts = {}
for key in CM_dict.keys():
    struct_type_counts[key] = df[f"{key}_struct_type"].value_counts()

struct_type_counts["materials_project"] = df["structure_type"].value_counts()

# Create a dataframe of the value counts
df_struct_type_counts = pd.DataFrame(struct_type_counts)
df_struct_type_counts.head()

# Compute the accuracy of the predictions
for key in CM_dict.keys():
    print(f'{key} accuracy = {df[f"{key}_same_struct_type"].mean():.3f}')

sys.exit(0)

# Bar plot of the results

sns.set(context="paper", font_scale=3)
sns.set_style(style='white')
ax = df_struct_type_counts.plot.bar(figsize=(8,8), rot=0)
ax.set_ylabel("Number of structures")
ax.set_xlabel("Structure type")
#ax.set_title("Structure types of predicted structures")
plt.savefig("structure_types.pdf", dpi=300, bbox_inches="tight")
plt.show()


# ### Alternate result visualisations
# 
# The averages and the barplot are not necessarily the most efficient way for communicating the results. 
# We can use confusion matrices as well to show off the specific class predictions

fig, axes = plt.subplots(4, 2, figsize=(18, 24), sharex="all", sharey="all")
class_labels = list(df["structure_type"].unique())
for key, ax in zip(CM_dict.keys(), axes.flatten()):
    true_label = df["structure_type"].values
    pred_label = df[f"{key}_struct_type"]
    pred_label.fillna("None", inplace=True)
    pred_label = pred_label.values
    # Add MCC:
    mc = mcc(true_label, pred_label)
    print(f'{key} mcc = {mc}')
    # Confusion mx
    cm = confusion_matrix(true_label, pred_label, labels=class_labels)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=class_labels
    ).plot(include_values=True, cmap="Blues", ax=ax)
    ax.set_title(f"{key}", fontweight="bold")

fig.suptitle("Confusion Matrices")
plt.tight_layout()
plt.savefig("confusion_matrix.pdf", bbox_inches="tight")
plt.show()

sys.exit(0)
# ### Generating tables
# The following section exists just for outputting tables of the formulas with their chosen templates to tex tables as well as a csv file.

# Set up table for outputting compositions
columns_to_keep = [
    "formula_pretty",
    "material_id",
    "oliynyk_formula",
    "matscholar_formula",
    "mat2vec_formula",
    "magpie_mat2vec_formula",
    "leaf_formula",
   "leaf200magpie_formula",
    "random_200_formula",
    "skipatom_formula",
    "magpie_formula",
    "megnet16_formula",
    "hautier_formula",
]

# Filter unrelated columns
comp_pred_df = df[columns_to_keep]

# Concatenate the material project ids and formula
comp_pred_df["target_formula"] = [
    f"{comp} ({comp_to_mpid[comp]})" for comp in comp_pred_df["formula_pretty"]
]

for key in CM_dict.keys():
    comp_pred_df[f"{key}_template"] = [
        f"{comp} ({comp_to_mpid[comp]})" if comp else None
        for comp in comp_pred_df[f"{key}_formula"]
    ]
comp_pred_df.drop(columns=columns_to_keep, inplace=True)
comp_pred_df.head()

sys.exit(0)

comp_pred_df.to_csv("formula_templates.csv", index=False)

# Create two dataframes to output to latex tables

latex_df1 = comp_pred_df[
    [
        "target_formula",
        "oliynyk_template",
        "matscholar_template",
        "mat2vec_template",
        "random_200_template",
    ]
]

latex_df2 = comp_pred_df[
    [
        "target_formula",
        "skipatom_template",
        "magpie_template",
        "megnet16_template",
        "hautier_template",
    ]
]

latex_df1.to_latex(
    "formula_templates1.tex",
    index=False,
    label="stab1",
    longtable=True,
    caption="Table of the target formulas from the Materials project and the template materials used to predict the structure of the target material. The template materials can be considered the most similar under each representation.",
)

latex_df2.to_latex(
    "formula_templates2.tex",
    index=False,
    label="stab2",
    longtable=True,
    caption="Table of the target formulas from the Materials project and the template materials used to predict the structure of the target material. The template materials can be considered the most similar under each representation.",
)

