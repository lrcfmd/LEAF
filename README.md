# LEAFs
Local Environment-induced Atomic Features (LEAFs) are derived from comprehensive learning of the local structures of chemical elements in inorganic crystals as described in 

A. Vasylenko et. al. "Learning Atoms from Crystal Structures", https://doi.org/10.48550/arXiv.2408.02292

LEAFs have been proven useful as  
1. Structurally-informed similarity measure for compositions.
2. Probability measure of elemental substitutions.
3. Featurization for elements and compositions.

We recommend installation via pip

```
pip install LEAFeatures
```

For local installation run

```
python setup.py install
```

## Usage
For simple usage initiate an object with its compositional formula

```python
> from LEAFeatures.leafs import Leaf
> x = Leaf("CaTiO3")
```

Calculate the distance to a second object with the `similarity` method, 

```python
> x.similarity("SrTiO3")
9.995487
```

which returns a number from 1 to 10, with ascending similarity. 

### Note,
the distance is calculated as cosine similarity between 59-dimensional elemental vectors. Hence, many distances > 9 and digits after decimal point are of determining importance. E.g.,
```python
> Leaf("CaTiO3").similarity("SrSiO3")
9.982714
> Leaf("CaTiO3").similarity("SrTiO3")
9.995487
```

Calculate the substitution probability with the `sub_probability` method. 

```python
> x.sub_probability("SrSiO3")
['Sr2+', 'Ti4+', 'O2-'] ['Ca2+', 'Ti4+', 'O2-'] [0.005256, 0.007006, 0.006997]
2.57654281392e-07
```

This returns the oxidation states derived assuming a charge balance, pairwise elemental subsitution probabilities (i.e. Sr -> Ca and Si -> Ti) ranging from 0 to 1, and their product.
Oxidation states of the candidate substitutions can be input explicitly:

```python
> x.sub_probability("Sr2TiO5", oxidations={'Sr':2, 'Ti':3, 'O':-2})
```

Because the probabilites product decreases for multi-element substitutions, comparing multi-element substitutions with the best possible substitutions for the same number of elements is advised.

```python
> x.best_substitutions(top=5)
Ca2+ [('Ca', 0.007486), ('Sr', 0.005256), ('K', 0.005169), ('Na', 0.005038), ('Eu', 0.004748)]
Ti4+ [('Ti', 0.007006), ('Sc', 0.005946), ('Zr', 0.005772), ('Mg', 0.005232), ('Mn', 0.005046)]
O2- [('O', 0.006997), ('F', 0.006419), ('N', 0.005402), ('Cl', 0.005153), ('C', 0.004748)]
Best probability: 3.6697107125199996e-07
```

This lists best top=10 substitution probabilities and the corresponding elements, by default for all elements in the composition, and the best value for 3-element substitution.
The best substitutions for the selected elements can be specified.

```python
> z.best_substitutions(['Ca2+','Ti4+'], top=10)
Ca2+ [('Ca', 0.007486), ('Sr', 0.005256), ('K', 0.005169), ('Na', 0.005038), ('Eu', 0.004748), ('Y', 0.00446), ('Rb', 0.004435), ('Ba', 0.004227), ('Mg', 0.004051), ('La', 0.003906)]
Ti4+ [('Ti', 0.007006), ('Sc', 0.005946), ('Zr', 0.005772), ('Mg', 0.005232), ('Mn', 0.005046), ('Nb', 0.004739), ('Cr', 0.004675), ('Y', 0.004629), ('V', 0.004548), ('Ta', 0.004236)]
Best probability: 5.2446916e-05
```

The LEAFs representation for compositions can be accessed.

```python
> z.representation
> Leaf('CaTiO3').representation
[6.57610834e-01 2.85473622e-01 4.96881850e-01 7.70918750e-01
 7.97946231e-01 4.61860970e-01 4.49522401e-01 4.86912308e-01
 4.66420565e-01 4.48792432e-01 4.28762788e-01 5.17797120e-01
 5.44486303e-01 4.65229431e-01 2.51304365e-01 3.44853466e-01
 4.01373918e-01 4.96830716e-01 9.00359667e-01 6.10222815e-01
 3.40766216e-01 4.43959061e-01 1.45974281e-01 5.54307962e-01
 2.14342638e-01 2.44581614e-01 2.90158378e-01 2.24863698e-01
 2.68514251e-01 3.07195609e-01 1.62546708e-01 1.77402081e-01
 2.23923315e-01 6.41999013e-01 1.72413736e-01 2.27852236e-01
 2.89814108e-01 6.60000000e+01 3.11000000e+02 1.35943200e+02
 3.22040000e+03 5.40000000e+01 1.40000000e+01 5.34000000e+02
 1.28600000e+01 1.00000000e+01 1.20000000e+01 2.00000000e+00
 0.00000000e+00 2.40000000e+01 0.00000000e+00 6.00000000e+00
 8.00000000e+00 0.00000000e+00 1.40000000e+01 8.17750000e+01
 0.00000000e+00 2.25333333e-05 4.55000000e+02]
```

To re-compute the LEAFs from the set of CIFs files, e.g.,
if the features are required for the specific subset of inorganic materials,
provide a list or a directory with CIFs, e.g. './testcifs':

```python

> from LEAFeatures.leafs import CreateLeaf
> newleaf = CreateLeaf()
> myleafs = newleaf.generate_from_cifs('./testcifs', mode='dir')
>
> # use newly created LEAFs:
> newx = Leaf("MgO", data=myleafs) 
> newx.representation
array([2.5875000e-01, 4.0000000e-03, 8.7250000e-02, 3.4150000e-01,
       1.0700000e-01, 1.5250000e-02, 1.0000000e-03, 1.0000000e-03,
       1.0000000e-03, 9.1462500e-01, 5.5662500e-01, 1.3344375e+00,
       1.3321250e+00, 1.0133750e+00, 3.8000000e-02, 7.5250000e-02,
       6.9750000e-02, 3.2500000e-01, 7.8912500e-01, 6.6725000e-01,
       6.2100000e-01, 7.6412500e-01, 1.0000000e-03, 1.0000000e-03,
       2.4500000e-02, 3.9500000e-02, 3.4875000e-02, 1.0000000e-03,
       1.0000000e-03, 1.0000000e-03, 2.8375000e-02, 2.8250000e-02,
       1.8500000e-02, 1.7675000e-01, 6.1937500e-02, 6.1875000e-02,
       8.5625000e-02])
