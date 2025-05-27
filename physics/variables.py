import numpy as np


c6_up, c6_dn = +18.7574, -18.7574
cH_up, cH_dn = +40, -40
c6_vals = [c6_up, c6_dn]
cH_vals = [cH_up, cH_dn]

eft_terms = [
    [1, 0, 0],
    [2, 0, 0],
    [3, 0, 0],
    [4, 0, 0],
    [0, 1, 0],
    [0, 2, 0],
    [0, 0, 1],
    [0, 0, 2],
    [1, 1, 0],
    [1, 0, 1],
    [2, 1, 0],
    [2, 0, 1],
    [0, 1, 1],
]

c6_degree = max([ctup[0] for ctup in eft_terms])
ct_degree = max([ctup[1] for ctup in eft_terms])
cg_degree = max([ctup[2] for ctup in eft_terms])