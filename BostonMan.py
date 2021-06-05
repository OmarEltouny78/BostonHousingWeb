import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import shap
import xgboost as xgb

st.write("""
# Boston House Price Prediction App
This app predicts the **Boston House Price**!
""")
st.write('---')

boston = load_boston()

X_bos_pd = pd.DataFrame(boston.data, columns=boston.feature_names)
Y_bos_pd = pd.DataFrame(boston.target)

st.sidebar.header('Specify Input Parameters')


def user_input_features():
    CRIM = st.sidebar.slider('CRIM', float(X_bos_pd.CRIM.min()), float(X_bos_pd.CRIM.max()),
                             float(X_bos_pd.CRIM.mean()))
    ZN = st.sidebar.slider('ZN', float(X_bos_pd.ZN.min()), float(X_bos_pd.ZN.max()), float(X_bos_pd.ZN.mean()))
    INDUS = st.sidebar.slider('INDUS', float(X_bos_pd.INDUS.min()), float(X_bos_pd.INDUS.max()),
                              float(X_bos_pd.INDUS.mean()))
    CHAS = st.sidebar.slider('CHAS', float(X_bos_pd.CHAS.min()), float(X_bos_pd.CHAS.max()),
                             float(X_bos_pd.CHAS.mean()))
    NOX = st.sidebar.slider('NOX', float(X_bos_pd.NOX.min()), float(X_bos_pd.NOX.max()), float(X_bos_pd.NOX.mean()))
    RM = st.sidebar.slider('RM', float(X_bos_pd.RM.min()), float(X_bos_pd.RM.max()), float(X_bos_pd.RM.mean()))
    AGE = st.sidebar.slider('AGE', float(X_bos_pd.AGE.min()), float(X_bos_pd.AGE.max()), float(X_bos_pd.AGE.mean()))
    DIS = st.sidebar.slider('DIS', float(X_bos_pd.DIS.min()), float(X_bos_pd.DIS.max()), float(X_bos_pd.DIS.mean()))
    RAD = st.sidebar.slider('RAD', float(X_bos_pd.RAD.min()), float(X_bos_pd.RAD.max()), float(X_bos_pd.RAD.mean()))
    TAX = st.sidebar.slider('TAX', float(X_bos_pd.TAX.min()), float(X_bos_pd.TAX.max()), float(X_bos_pd.TAX.mean()))
    PTRATIO = st.sidebar.slider('PTRATIO', float(X_bos_pd.PTRATIO.min()), float(X_bos_pd.PTRATIO.max()),
                                float(X_bos_pd.PTRATIO.mean()))
    B = st.sidebar.slider('B', float(X_bos_pd.B.min()), float(X_bos_pd.B.max()), float(X_bos_pd.B.mean()))
    LSTAT = st.sidebar.slider('LSTAT', float(X_bos_pd.LSTAT.min()), float(X_bos_pd.LSTAT.max()),
                              float(X_bos_pd.LSTAT.mean()))
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

st.header('Specified Input parameters')
st.write(df)
st.write('---')

xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10,
                          n_estimators=10)

X_train, X_test, Y_train, Y_test = train_test_split(X_bos_pd, Y_bos_pd, test_size=0.2)

xg_reg.fit(X_train, Y_train)
preds = xg_reg.predict(X_test)

st.header('Prediction of MEDV')
st.write(preds)
st.write('---')

explainer = shap.TreeExplainer(xg_reg)
shap_values = explainer.shap_values(X_bos_pd)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X_bos_pd)
st.pyplot(bbox_inches='tight')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X_bos_pd, plot_type="bar")
st.pyplot(bbox_inches='tight')
st.set_option('deprecation.showPyplotGlobalUse', False)