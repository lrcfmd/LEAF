# first run for all room T data , featurised with Leaf for Li and cat - concatenated
# second run for reduced data: 236 low; 60 high to highlight features conducive to low conductivity

# first selected top 9 features with 'entropy' criterion: acc 75 f1 68%

first = ['cat_pentagonal planar CN_5', 'Li_pentagonal pyramidal CN_6',
 'Li_trigonal non-coplanar CN_3', 'Li_rectangular see-saw-like CN_4',
 'Li_q6 CN_12', 'cat_trigonal bipyramidal CN_5',
 'Li_pentagonal bipyramidal CN_7', 'Li_octahedral CN_6',
 'Li_hexagonal pyramidal CN_7']

# second selected top 9 features with 'entropy' criterion: acc 84 f1 53%
second =['cat_linear CN_2', 'Li_rectangular see-saw-like CN_4',
 'Li_hexagonal pyramidal CN_7','Li_octahedral CN_6',
 'cat_trigonal pyramidal CN_4', 'Li_trigonal planar CN_3',
 'cat_bent 120 degrees CN_2', 'Li_q2 CN_10',
 'Li_pentagonal bipyramidal CN_7']

# the intersection - features conducive to low conductivity:

print (set(first).intersection(set(second)))
