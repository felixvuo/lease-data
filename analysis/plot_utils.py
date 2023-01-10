import numpy as np

## FIX names of encodings in portfolio
# on the server we have: ggt ggth gmto gpw lpw mdd rggt swc tree
# in the PB(AMO) paper: 
def fixEncs(portfolio_string):
    fromto = (('TREE', 'Tree'), ('GPW', 'GGPW'), ('SWC','GSWC'), ('LPW', 'GLPW'),
              ('GGTH','+ggt+'), ('RGGT', '+rggt+'), ('GGT','GGTd'),
              ('+ggt+','GGT'), ('+rggt+','RGGT') )
    p_ = portfolio_string.upper()
    for f,t in fromto:
        p_ = p_.replace(f,t)
    return p_

fixEncsV = np.vectorize(fixEncs)

fig_width_in = 4.5
seaborn_opts = dict(
    context="paper",
    style="ticks",
    font_scale=0.64,
    rc={
        "font.family": "sans-serif",
        "lines.linewidth"  : 1,
        "axes.linewidth" : 0.5,
        "xtick.major.width" : 0.5,
        "ytick.major.width" : 0.5,
        "xtick.major.size" : 3,
        "ytick.major.size" : 3,
    }
)
