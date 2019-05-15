import matplotlib.pyplot as plt


import seaborn as sns
sns.set_style('ticks')
sns.set_context('notebook', font_scale=1.5)


colors = sns.color_palette()
def despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False):
    plt.tight_layout()
    sns.despine(fig, ax, top, right, left, bottom, offset, trim)
