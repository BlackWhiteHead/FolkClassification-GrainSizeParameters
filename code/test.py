import pandas as pd
from plot_evaluation import *
from predict_xgboost import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn import metrics

df_in = pd.read_csv('example data.csv')
df = predict_xgb(df_in)
df.to_csv('output/results.csv', index=False)

cm1 = confusion_matrix(df['code_ng'], df['code_ng_pred'])
df_ng = df.loc[(df['code_ng'] == 1) & (df['code_ng_pred'] == 1), ['Folk_class', 'Folk_class_pred']]
cm2 = confusion_matrix(df_ng['Folk_class'], df_ng['Folk_class_pred'])
label1 = np.unique(df['code_ng_pred'])
label2 = np.unique(df_ng['Folk_class_pred'])
print(metrics.classification_report(df_ng['Folk_class'], df_ng['Folk_class_pred']))

fig1, ax1 = plt.subplots(1, 1, figsize=[3.5, 3], dpi=300)
plot_confusion_matrix(cm1, label1, ax=ax1)
fig1.tight_layout()
fig1.savefig('output/figure1.jpg', bbox_inches='tight')

fig2, ax2 = plt.subplots(1, 1, figsize=[3.5, 3], dpi=300)
plot_confusion_matrix(cm2, label2, ax=ax2)
fig2.tight_layout()
fig2.savefig('output/figure2.jpg', bbox_inches='tight')

fig3, ax3 = plt.subplots(1, 1, figsize=[3.5, 3], dpi=300)
plot_ternary_Folk_B(df, 'Folk_class_pred', ax=ax3, density=False)
fig3.savefig('output/figure3.jpg', bbox_inches='tight')

df['Error'] = df[['Folk_class', 'Folk_class_pred']].apply(
    lambda x: False if x.Folk_class == x.Folk_class_pred else True, axis=1)
fig4, ax4 = plt.subplots(1, 1, figsize=[3.5, 3], dpi=300)
plot_ternary_Folk_B(df.loc[df['Error'] == True, :], 'Error', ax=ax4, density=True)
fig4.savefig('output/figure4.jpg', bbox_inches='tight')