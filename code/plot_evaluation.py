import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from scipy.stats import gaussian_kde
import ternary


def plot_confusion_matrix(cm, labels, ax):
    fontsize = 7
    plt.rc('font', family='Arial', size=fontsize)
    plt.tick_params(labelsize=fontsize)

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)

    tick_major = np.arange(cm.shape[0])
    x, y = np.meshgrid(tick_major, tick_major)
    for xi, yi in zip(x.flatten(), y.flatten()):
        val = cm[yi][xi]
        if val > 0:
            ax.text(xi, yi, val, color='red', va='center', ha='center')
    ax.set_xlabel('Predicted label', fontsize=fontsize)
    ax.set_ylabel('Actual label', fontsize=fontsize)
    ax.set(xticks=tick_major, yticks=tick_major,
           xticklabels=labels, yticklabels=labels)

    tick_minor = np.arange(cm.shape[0] + 1) - 0.5
    ax.set_xticks(tick_minor, minor=True)
    ax.set_yticks(tick_minor, minor=True)
    ax.tick_params(which='both', bottom=False, left=False)
    ax.grid(True, which='minor', linestyle='-', lw=1.5)

    linewidth = 1
    ax.spines['top'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)

    cax = plt.colorbar(im, ax=ax, shrink=0.9)
    cax.ax.tick_params(width=linewidth)
    cax.outline.set_linewidth(linewidth)

    return


def plot_ternary_Folk_B(df, ylabel, ax, density=False):
    fontsize = 7
    plt.rc('font', family='Arial', size=fontsize)
    ptsize = 10
    linewidth = 1

    fig, tax = ternary.figure(ax=ax, scale=100)

    tax.boundary(linewidth=linewidth)
    tax.horizontal_line(90, linewidth=linewidth, color='blue')
    tax.horizontal_line(50, linewidth=linewidth, color='blue')
    tax.horizontal_line(10, linewidth=linewidth, color='blue')
    p1 = (10 / 3, 90, 20 / 3)
    p2 = (100 / 3, 0, 200 / 3)
    tax.line(p1, p2, linewidth=linewidth, color='blue')
    p1 = (20 / 3, 90, 10 / 3)
    p2 = (200 / 3, 0, 100 / 3)
    tax.line(p1, p2, linewidth=linewidth, color='blue')

    tax.right_corner_label("Clay")
    tax.top_corner_label("Sand")
    tax.left_corner_label("Silt")

    labels = df[ylabel].value_counts().index
    cmap = mpl.cm.get_cmap('plasma', len(labels))
    colors = cmap(range(len(labels)))
    markers = ['.', ',', 'v', '+', 'o', '*', '<', '>', 'D', '1', 's', '2', 'h']
    df_show = df[df['Sand'] + df['Silt'] + df['Clay'] > 99.9]

    if density:
        x = df_show['Sand'].values
        y = df_show['Silt'].values
        z = df_show['Clay'].values
        xy = np.vstack([x, y])
        c = gaussian_kde(xy)(xy)

        idx = c.argsort()
        x, y, z, c = x[idx], y[idx], z[idx], c[idx]

        cb_kwargs = {"shrink": 1.0,
                     "orientation": "horizontal",
                     "fraction": 0.1,
                     "pad": 0.01,
                     "aspect": 30, }

        s = tax.scatter(tuple(zip(z, x, y)), vmax=max(c), colormap=plt.cm.plasma, colorbar=False,
                        c=c, cmap=plt.cm.plasma, s=ptsize, edgecolor=None, linewidths=0)
        cb = s.figure.colorbar(s.collections[0], **cb_kwargs)
        cb.ax.tick_params(width=linewidth)
        cb.outline.set_linewidth(linewidth)
    else:
        for label, color in zip(labels, colors):
            tax.scatter(tuple(df_show.loc[df_show[ylabel] == label, ['Clay', 'Sand', 'Silt']].values),
                        marker=random.choice(markers), color=color, label=label, s=ptsize, edgecolor=None, linewidths=0)
        tax.legend(loc=1, bbox_to_anchor=(1, 1, 0.01, 0.01), markerscale=2)

    tax.annotate(text='S', position=(3, 92))
    tax.annotate(text='zS', position=(4, 65))
    tax.annotate(text='mS', position=(14, 65))
    tax.annotate(text='cS', position=(25, 65))
    tax.annotate(text='sZ', position=(10, 28))
    tax.annotate(text='sM', position=(32, 28))
    tax.annotate(text='sC', position=(56, 28))
    tax.annotate(text='Z', position=(15, 3))
    tax.annotate(text='M', position=(45, 3))
    tax.annotate(text='C', position=(80, 3))

    ticks = list(np.arange(0, 101, 20))
    tax.ticks(ticks=ticks, axis='lbr', linewidth=1, multiple=1, offset=0.02, clockwise=False, fontsize=fontsize)
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()
