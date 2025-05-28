import numpy as np

c6_val = +18.7574
cH_val = -40

ct_val = +1.27167
ctH_val = +20

cg_val = +1.35156
cHG_val = +0.1

c6_to_cH  = cH_val/c6_val
ct_to_ctH = ctH_val/ct_val
cg_to_cHG = cHG_val/cg_val

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