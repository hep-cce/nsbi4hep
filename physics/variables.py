import numpy as np


c6_up, c6_dn = +18.7574, -18.7574
cH_up, cH_dn = +40, -40
c6_vals = [c6_up, c6_dn]
cH_vals = [cH_up, cH_dn]

to_smeft = c6_up/cH_up
from_smeft = cH_up/c6_up

cHbox_up, cHbox_dn = 0.05,-0.02

# c6_space = np.linspace(c6_dn, c6_up, 201)
# cHbox_space = np.linspace(*)

# c6_sm = 0.0
# cHbox_sm = 0.0

# i_c6_sm = np.where(c6_space==c6_sm)[0][0]
# i_cHbox_sm = np.where(np.round(cHbox_space,5)==cHbox_sm)[0][0]

# c6_asimov = 0.0
# cHbox_asimov = 0.0

# i_c6_asimov = np.where(c6_space==c6_asimov)[0][0]
# i_cHbox_asimov = np.where(np.round(cHbox_space,5)==cHbox_asimov)[0][0]