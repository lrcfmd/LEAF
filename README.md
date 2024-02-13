# LEAFs
Local Environment-induced Atomic Features (LEAFs) are derived from comprehensive learning of the local structures of chemical elements in inorganic crystals as described in 

A. Vasylenko et. al. "Learning Atoms from Crystal Structures".

LEAFs have been proven useful as  
1. Structurally-informed similarity measure for compositions.
2. Probability measure of elemental substitutions.
3. Featurization for elements and compositions.

We recommend installation via pip

```
pip install LEAFs
```

## Usage
For simple usage initiate an object with its compositional formula

```python
> from LEAFs import LEAF
> x = LEAF("CaTiO3")
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
> LEAF("CaTiO3").similarity("SrSiO3")
9.982714
> LEAF("CaTiO3").similarity("SrTiO3")
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

This lists best top=5 substitution probabilities and the corresponding elements, by default for all elements in the composition, and the best value for 3-element substitution.
The best substitutions for the selected elements can be specified, e.g., 

```python
> z.best_substitutions(['Ca2+','Ti4+'], top=10)
Ca2+ [('Ca', 0.007486), ('Sr', 0.005256), ('K', 0.005169), ('Na', 0.005038), ('Eu', 0.004748), ('Y', 0.00446), ('Rb', 0.004435), ('Ba', 0.004227), ('Mg', 0.004051), ('La', 0.003906)]
Ti4+ [('Ti', 0.007006), ('Sc', 0.005946), ('Zr', 0.005772), ('Mg', 0.005232), ('Mn', 0.005046), ('Nb', 0.004739), ('Cr', 0.004675), ('Y', 0.004629), ('V', 0.004548), ('Ta', 0.004236)]
Best probability: 5.2446916e-05

```
