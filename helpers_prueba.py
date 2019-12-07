import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.preprocessing import PolynomialFeatures as polynom
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, classification_report, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score



def coefplot(model, varnames=True, intercept=False, fit_stats=True, figsize=(7, 12)):
    """
    coefplot - Visualize coefficient magnitude and approximate frequentist significance from a model.
    
    @parameters:
        - model: a `statsmodels.formula.api` class generated method, which must be already fitted.
        - varnames: if True, y axis will contain the name of all of the variables included in the model. Default: True
        - intercept: if True, coefplot will include the $\beta_{0}$ estimate. Default: False.
        - fit_stats: if True, coefplot will include goodness-of-fit statistics. Default: True.
        
    @returns:
        - A `matplotlib` object.
    """
    if intercept is True:
        coefs = model.params.values
        errors = model.bse
        if varnames is True:
            varnames = model.params.index
    else:
        coefs = model.params.values[1:]
        errors = model.bse[1:]
        if varnames is True:
            varnames = model.params.index[1:]
            
    tmp_coefs_df = pd.DataFrame({'varnames': varnames, 'coefs': coefs,'error_bars': errors})
    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(y=tmp_coefs_df['varnames'], x=tmp_coefs_df['coefs'], 
                xerr=tmp_coefs_df['error_bars'], fmt='o', 
                color='slategray', label='Estimated point')
    ax.axvline(0, color='tomato', linestyle='--', label='Null Effect')
    ax.set_xlabel(r'$\hat{\beta}$')
    fig.tight_layout()
    plt.legend(loc='best')
    
    if fit_stats is True:
        if 'linear_model' in model.__module__.split('.'):
            plt.title(r'R$^{2}$' + "={0}, f-value={1}, n={2}".format(round(model.rsquared, 2),
                                                                     round(model.f_pvalue, 3),
                                                                     model.nobs))
        elif 'discrete_model' in model.__module__.split('.'):
            plt.title("Loglikelihood = {0}, p(ll-Rest)={1}, n={2}".format(round(model.llf, 2),
                                                                          round(model.llr_pvalue, 3),
                                                                          model.nobs))

def saturated_model(df, dependent, estimation=smf.ols,fit_model=True):
    """
    saturated_model - returns a saturated model

    @parameters:
        - df: pandas.core.frame.DataFrame.
        - dependent: String. Name of the variable that wants to be predicted.
        - estimation: Method. smf estimator
        - fit_model: Bool. If the model wants to be returned with fith done or not.

    @returns:
        - A `smf` model.

    """
    tmp_vars = "+".join(df.columns.drop(dependent))
    tmp_model = estimation(dependent+ '~ '+ tmp_vars, df)
    if fit_model is True:
        tmp_model = tmp_model.fit()
    
    return tmp_model

def concise_summary(mod, estimation="ols", print_fit=True):
    """
    concise_summary - shows coef, std.error and t/z of given model

    @parameters:
        - mod: statsmodels.discrete.discrete_model.BinaryResultsWrapper.
        - estimation: String. Name of the smf estimation. Possible values: "ols", "logit".
        - print_fit: Bool. If goodness of fit statistics wants to be added to de output or not.

    @returns:
        - A String output.

    """
    #guardamos los parámetros asociados a estadísticas de ajuste
    fit = pd.DataFrame({'Statistics': mod.summary2().tables[0][2][2:],
                       'Value': mod.summary2().tables[0][3][2:]})
    # guardamos los parámetros estimados por cada regresor.
    if estimation is "ols":
        estimates = pd.DataFrame(mod.summary2().tables[1].loc[:, ['Coef.', 'Std.Err.', 't']])
    elif estimation is "logit":
        estimates = pd.DataFrame(mod.summary2().tables[1].loc[:, ['Coef.', 'Std.Err.', 'z']])
        
    # imprimir fit es opcional
    if print_fit is True:
        print("\nGoodness of Fit statistics\n", fit)
    
    print("\nPoint Estimates\n\n", estimates)

def plot_roc_curve(model, xtest_std, ytest):
    """
    plot_roc_curve - Plots ROC curve of given model.
    
    @parameters:
        - model: a `statsmodels.formula.api` class generated method, which must be already fitted.
        - xtest_std: numpy.ndarray. A standarized "x" sample, different from the one used in the model. Object.
        - ytest: pandas.core.series.Series. The sample to compare with the generated predicted values. Object.
        
    @returns:
        - A `matplotlib` object.
    """
    yhat = model.predict_proba(xtest_std)[:, 1]
    false_positive, true_positive, threshold = roc_curve(ytest, yhat)
    # Plot ROC curve
    plt.title('Curva ROC')
    plt.plot(false_positive, true_positive, lw=1)
    plt.plot([0, 1], ls="--", lw=1)
    plt.plot([0, 0], [1, 0] , c='limegreen', lw=3), plt.plot([1, 1] , c='limegreen', lw=3)
    plt.ylabel('Verdaderos Positivos')
    plt.xlabel('Falsos Positivos');