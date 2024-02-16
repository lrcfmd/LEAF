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
The best substitutions for the selected elements can be specified.

```python
> z.best_elemental_subs(['Ca2+','Ti4+'], top=10)
Ca2+ [('Ca', 0.007486), ('Sr', 0.005256), ('K', 0.005169), ('Na', 0.005038), ('Eu', 0.004748), ('Y', 0.00446), ('Rb', 0.004435), ('Ba', 0.004227), ('Mg', 0.004051), ('La', 0.003906)]
Ti4+ [('Ti', 0.007006), ('Sc', 0.005946), ('Zr', 0.005772), ('Mg', 0.005232), ('Mn', 0.005046), ('Nb', 0.004739), ('Cr', 0.004675), ('Y', 0.004629), ('V', 0.004548), ('Ta', 0.004236)]
Best probability: 5.2446916e-05
```

The LEAFs representation for compositions can be accessed.

```python
> z.representation
> LEAF('CaTiO3').representation
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
Find top-N similar compositions and corresponding structure types in ICSD.

```python
> z.best_icsd_similarity(top=10, drop_self_duplicates=True, unique_structure_type=True)
Top 10 most similar reported compositions:
         composition cosine similarity            structure type       crystal system
   Ca0.99Ti1Cu0.01O3   1.000000              Perovskite#GdFeO3         orthorhombic
   Ca0.99Ti1Zn0.01O3   1.000000    Perovskite#(Ca,Li)(Zr,Ta)O3         orthorhombic
  Ca1Ti0.9Fe0.1O2.95   0.999998              Perovskite#SrZrO3           tetragonal
     Ca1Ti0.8Fe0.2O3   0.999993                Na(Cl,Br)O3(P1)            triclinic
   Ca1Ti0.8Fe0.2O2.9   0.999992              Perovskite#CaTiO3                cubic
Ca0.999Ti0.805Fe0.201O2.899   0.999992                            nan         orthorhombic
 Ca9Y1Nb0.28V6.72O28   0.999988 Whitlockite#\xce\xb2-Ca3(PO4)2             trigonal
        Na0.912Ti2O4   0.999982                 CaV2O4#CaFe2O4         orthorhombic
           Ca4Ti3O10   0.999968                      Ca4Mn3O10         orthorhombic
             Ca2V2O7   0.999967                        Ca2V2O7            triclinic
```
