import xgboost as xgb
import pandas as pd
import math


def predict_xgb(df_in):
    df = df_in.copy()
    cols_input = ['Mz', 'Sk', 'Ku', 'Sigma']
    dinput = xgb.DMatrix(df[cols_input])

    bst = xgb.Booster()
    bst.load_model('model/xgb_2.model')
    ypred = bst.predict(dinput)
    df['code_ng_pred'] = pd.Series(ypred, index=df.index, dtype='int')

    dinput_ng = xgb.DMatrix(df.loc[df['code_ng_pred'] == 1, cols_input])
    bst = xgb.Booster()
    bst.load_model('model/xgb_cls.model')
    ypred = bst.predict(dinput_ng)
    df['code_class_pred'] = pd.Series(ypred, index=df[df['code_ng_pred'] == 1].index)

    code_class = pd.read_csv('model/code.csv', header=None)
    code_class.columns = ['Folk_class', 'class_val']
    code_dict = dict(zip(code_class['class_val'], code_class['Folk_class']))
    df['Folk_class_pred'] = df['code_class_pred'].apply(lambda x: code_dict[x] if not math.isnan(x) else x)

    df = df.drop('code_class_pred', axis=1)
    df['code_ng'] = df['Gravel'].apply(lambda x: 0 if x > 0 else 1)
    cols_result = ['Folk_class_pred', 'code_ng', 'code_ng_pred']
    df_result = df[cols_result].copy()
    df = df.drop(cols_result, axis=1)
    df[cols_result] = df_result

    return df
