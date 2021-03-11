#Copyright notice

#Please remember that this package is provided to you by www.deepcreditrisk.com and its authors Daniel Roesch and Harald Scheule. The package is protected by copyright. You are not permitted to re-use for commercial purposes without permission of the copyright owner. Improper or illegal use may lead to prosecution for copyright infringement. 

#The module provides the package references for the functions used in book 'Roesch, D./ Scheule, H.: Deep Credit Risk - Machine learning in Python, 2020, Amazon':

#packages and basic settings
#versions
#dataprep
#woe
#validation
#resolutionbias

#packages and basic settings

import IPython
import math
import matplotlib.pyplot as plt
import joblib
import numpy as np
import numpy.matlib
import pandas as pd
import pickle
import pydot
import pylab
import random
import scipy 
import scipy.cluster.hierarchy as shc
import scipy.stats as stats
import sklearn
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys
import tabulate

from numpy import linalg
from pylab import *
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.integrate import *
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform 
from scipy.stats import *
from scipy.stats import binom, beta, expon, mvn, randint as sp_randint, shapiro, ttest_ind, bernoulli
from sklearn import metrics
from sklearn.calibration import calibration_curve
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier, VotingClassifier, BaggingRegressor, RandomForestRegressor, AdaBoostRegressor,GradientBoostingRegressor, VotingRegressor
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, RFE, RFECV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, SGDClassifier
from sklearn.metrics import classification_report,confusion_matrix, r2_score, make_scorer, mean_squared_error, mean_absolute_error,roc_curve,accuracy_score,roc_auc_score,brier_score_loss, precision_score, recall_score,f1_score, log_loss
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV, train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import  KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

pd.set_option('display.max_rows',10)

# Change path here if data is saved in a different location
data = pd.read_csv('C:/TMP/dcr.csv')

import warnings; warnings.simplefilter('ignore')

# Function Versions    

def versions():
    pd.set_option('display.max_rows',30)
    if IPython.version_info==(7, 6, 1, ''): 
        iPython='7.6.1' 
    elif IPython.version_info!=(7, 6, 1, ''): 
        iPython='NaN'
    table = [

                          
                          ['Python', sys.version[:5], 'NaN'],
                          ['IPython', iPython, 'IPython'],
                          ['math', NaN, 'math'],
                          ['matplotlib.pyplot, pylab', matplotlib.__version__, 'plt'],
                          ['numpy', numpy.__version__, 'np'],
                          ['pandas', pd.__version__, 'pd'],        
                          ['pickle', pickle.format_version, 'pickle'],
                          ['random', NaN, 'random'],
                          ['scipy', scipy.__version__, 'scipy'],        
                          ['sklearn', sklearn.__version__, 'sklearn'],
                          ['statsmodels', sm.__version__, 'sm'],
                          ['tabulate', tabulate.__version__, 'tabulate'],

    ]
    table=pd.DataFrame(data=table)
    table.columns = ['Package', 'Version', 'Acronym']
    print(table)
    pd.set_option('display.max_rows',10)
    
# Function dataprep

def dataprep(data_in, depvar='default_time', splitvar='time', threshold=26):
    
    df=data_in.dropna(subset=['time', 'default_time','LTV_time', 'FICO_orig_time']).copy()
    
    # Economic features
    df.loc[:, 'annuity'] = ((df.loc[:,'interest_rate_time']/(100*4))*df.loc[:,'balance_orig_time'])/(1-(1+df.loc[:,'interest_rate_time']/(100*4))**(-(df.loc[:,'mat_time']-df.loc[:,'orig_time'])))
    df.loc[:,'balance_scheduled_time']  = df.loc[:,'balance_orig_time']*(1+df.loc[:,'interest_rate_time']/(100*4))**(df.loc[:,'time']-df.loc[:,'orig_time'])-df.loc[:,'annuity']*((1+df.loc[:,'interest_rate_time']/(100*4))**(df.loc[:,'time']-df.loc[:,'orig_time'])-1)/(df.loc[:,'interest_rate_time']/(100*4))
    df.loc[:,'property_orig_time'] = df.loc[:,'balance_orig_time']/(df.loc[:,'LTV_orig_time']/100)
    df.loc[:,'cep_time']= (df.loc[:,'balance_scheduled_time'] - df.loc[:,'balance_time'])/df.loc[:,'property_orig_time']

    df.loc[:,'equity_time'] = 1-(df.loc[:,'LTV_time']/100)

    df=df.dropna(subset=['time', 'cep_time', 'equity_time'])
    
    df.loc[:,'age'] = (df.loc[:,'time']-df.loc[:,'first_time']+1)
    df.loc[df['age'] >= 40, 'age'] = 40    
    df.loc[:,'age_1'] = df.loc[:,'time']-df.loc[:,'first_time']
    df.loc[df['age_1'] >= 39, 'age_1'] = 39
    df.loc[:,'age_1f'] = df.loc[:,'age_1']
    df.loc[df['age_1f'] <= 1, 'age_1f'] = 1
    df.loc[:,'age2'] = df.loc[:,'age']**2
    
    df['vintage'] = df.loc[:,'orig_time']
    df.loc[df['vintage'] < 0, 'vintage'] = 0
    df.loc[df['vintage'] >= 30, 'vintage'] = 30
    df.loc[:,'vintage2'] = df.loc[:,'vintage']**2
    
    df.loc[:,'state_orig_time'] = pd.Categorical(df.state_orig_time, ordered=False)

    if depvar=='default_time':
        df2 = df

        df2 = df2.loc[df2['state_orig_time'] != 'AL',:].copy()
        df2 = df2.loc[df2['state_orig_time'] != 'AK',:].copy()
        df2 = df2.loc[df2['state_orig_time'] != 'AR',:].copy()
        df2 = df2.loc[df2['state_orig_time'] != 'ND',:].copy()
        df2 = df2.loc[df2['state_orig_time'] != 'SD',:].copy()
        df2 = df2.loc[df2['state_orig_time'] != 'MT',:].copy()
        df2 = df2.loc[df2['state_orig_time'] != 'DE',:].copy()
        df2 = df2.loc[df2['state_orig_time'] != 'WV',:].copy()
        df2 = df2.loc[df2['state_orig_time'] != 'VT',:].copy()
        df2 = df2.loc[df2['state_orig_time'] != 'ME',:].copy()
        df2 = df2.loc[df2['state_orig_time'] != 'NE',:].copy()
        df2 = df2.loc[df2['state_orig_time'] != 'NH',:].copy()
        df2 = df2.loc[df2['state_orig_time'] != 'MS',:].copy()
        df2 = df2.loc[df2['state_orig_time'] != 'VI',:].copy()
        df2 = df2.loc[df2['state_orig_time'] != 'DC',:].copy()
        df2 = df2.loc[df2['state_orig_time'] != 'PR',:].copy()
        df2 = df2.loc[df2['state_orig_time'] != 'nan',:].copy() 
        
        # Splitting
        data_train = df2.loc[df2[splitvar] < threshold+1,:].copy()
        data_test = df2.loc[df2[splitvar] > threshold,:].copy()

        # PCA
        defaultrates_states_train = data_train.groupby(['time', 'state_orig_time'])['default_time'].mean().unstack(level=1).add_prefix('defaultrate_').fillna(0).reset_index(drop=False)
        defaultrates_states = df2.groupby(['time', 'state_orig_time'])['default_time'].mean().unstack(level=1).add_prefix('defaultrate_').fillna(0).reset_index(drop=False)
        
        scaler = StandardScaler().fit(defaultrates_states_train)
        defaultrates_states_train1 = scaler.transform(defaultrates_states_train)
        defaultrates_states1 = scaler.transform(defaultrates_states)

        pca = PCA()
        pca.fit(defaultrates_states_train1)  
        z_train = pca.transform(defaultrates_states_train1)
        z = pca.transform(defaultrates_states1)
        z_train = z_train[:,0:5]
        z = z[:,0:5]

        Z_train = pd.DataFrame(data=z_train, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5'])
        Z = pd.DataFrame(data=z, columns=['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5'])

        Z_train_1 = Z_train.shift(1).add_suffix('_1')
        Z_1 = Z.shift(1).add_suffix('_1')

        defaultrates_states_train2 = pd.concat([defaultrates_states_train['time'], Z_train_1], axis=1).dropna(subset=['PCA1_1']).copy()
        defaultrates_states2 = pd.concat([defaultrates_states['time'], Z_1], axis=1).dropna(subset=['PCA1_1']).copy() 

        data_train = pd.merge(data_train, defaultrates_states_train2, on='time')
        df3 = pd.merge(df2, defaultrates_states2, on='time')

        data_test = df3.loc[df3[splitvar] > threshold,:].copy()
        
        # Scaling
        X_train = data_train[['cep_time', 'equity_time', 'interest_rate_time', 'FICO_orig_time',  'gdp_time', 'PCA1_1','PCA2_1', 'PCA3_1','PCA4_1','PCA5_1']].dropna()
        X_test = data_test[['cep_time', 'equity_time', 'interest_rate_time', 'FICO_orig_time',  'gdp_time',  'PCA1_1','PCA2_1', 'PCA3_1','PCA4_1','PCA5_1']].dropna()
        
        
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        y_train = data_train['default_time'].values.reshape(-1,)
        y_test = data_test['default_time'].values.reshape(-1,)

        # Clustering
        n_clusters = 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=2, verbose=0)
        kmeans.fit(X_train_scaled)

        clusters_train =kmeans.predict(X_train_scaled)
        clusters_test = kmeans.predict(X_test_scaled)

        dummies_train = pd.get_dummies(clusters_train, drop_first=True, prefix='cluster')
        dummies_test = pd.get_dummies(clusters_test, drop_first=True, prefix='cluster')
        
        X_train_scaled = np.append(X_train_scaled, dummies_train, axis=1)
        X_test_scaled = np.append(X_test_scaled, dummies_test, axis=1)

        dummies = pd.concat([dummies_train, dummies_test], axis=0, ignore_index=True)
        dummies = dummies.reindex(data.index)

        df3 = pd.concat([df3, dummies], axis=1).dropna(subset=['id'])
        data_train = pd.concat([data_train, dummies_train], axis=1)
        dummies_test = dummies_test.reindex(data_test.index)
        data_test  = pd.concat([data_test,  dummies_test],  axis=1)
        
    if depvar=='lgd_time':
        
        # LGD dataprep
        df2 = df.query('default_time == 1').copy()    
        df3 = resolutionbias(df2,'lgd_time','res_time','time')
        
        df3.loc[df3['lgd_time'] <= 0, 'lgd_time'] = 0.0001
        df3.loc[df3['lgd_time'] >= 1, 'lgd_time'] = 0.9999

        # Splitting
        data_train =df3.loc[df3[splitvar] < threshold+1,:].copy()
        data_test =df3.loc[df3[splitvar] > threshold,:].copy()
        
        X_train = data_train[['cep_time', 'equity_time', 'interest_rate_time', 'FICO_orig_time', 'REtype_CO_orig_time', 'REtype_PU_orig_time', 'gdp_time']]
        X_test = data_test[['cep_time', 'equity_time', 'interest_rate_time', 'FICO_orig_time', 'REtype_CO_orig_time', 'REtype_PU_orig_time', 'gdp_time']]
    
        y_train = data_train['lgd_time'].values.reshape(-1,)
        y_test = data_test['lgd_time'].values.reshape(-1,)
        
        # Scaling
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        dummies_train = pd.get_dummies(data_train.state_orig_time, drop_first=True, prefix='state_orig_time')
        dummies_test = pd.get_dummies(data_test.state_orig_time, drop_first=True, prefix='state_orig_time')

        X_train_scaled = np.append(X_train_scaled, dummies_train, axis=1)
        X_test_scaled = np.append(X_test_scaled, dummies_test, axis=1)     
    
    return df3, data_train, data_test, X_train_scaled, X_test_scaled, y_train, y_test

# Function woe

def woe(data_in, target, variable, bins, binning):
    
    df = data_in
    df2 = data_in[[target, variable]].rename(columns={target: 'Target', variable: 'Variable'}).dropna()
    
    if binning == 'True':
       df2['key'] = pd.qcut(df2.Variable, bins, labels=False, duplicates='drop')
    if binning == 'False':
       df2['key'] = df2.Variable
    table = pd.crosstab(df2.key, df2.Target, margins= True)
    table = table.drop(['All'], axis=0)
    table = table.rename(columns={1: 'deft', 0: 'nondeft'}).reset_index(drop=False)

    table.loc[:, 'fracdeft'] = table.deft/np.sum(table.deft)
    table.loc[:, 'fracnondeft'] = table.nondeft/np.sum(table.nondeft)

    table.loc[:, 'WOE'] = np.log(table.fracdeft/table.fracnondeft)
    table.loc[:, 'IV'] = (table.fracdeft-table.fracnondeft)*table.WOE
    
    table.rename(columns={'WOE': variable}, inplace=True)
    table=table.add_suffix('_WOE')
    table.rename(columns={table.columns[0]: 'key' }, inplace = True)
    WOE = table.iloc[:, [0,-2]]
    
    df = pd.merge(df, df2.key, right_index=True, left_index=True)
      
    outputWOE = pd.merge(df, WOE, on='key').drop(['key'], axis=1)
    outputIV = pd.DataFrame(data={'name': [variable], 'IV': table.IV_WOE.sum()})
    
    return outputWOE, outputIV


# Function validation


def validation(fit, outcome , time, continuous=False):

    plt.rcParams['figure.dpi']= 300
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.rcParams.update({'font.size': 16})
    
    fitP=pd.DataFrame(data=fit)
    outcomeP=pd.DataFrame(data=outcome)
    timeP=pd.DataFrame(data=time)
    
    if isinstance(fit, pd.Series):
        fit=fit.values
    if isinstance(outcome, pd.Series):
        outcome=outcome.values
    if isinstance(time, pd.Series):
        time=time.values
    
    data_in = pd.concat([fitP, outcomeP, timeP], axis=1)
    data_in.columns = ['fit', 'outcome', 'time']
    means = data_in.groupby('time')[['fit', 'outcome']].mean().reset_index(drop=False)
  
    data_in['outcomeD']=data_in.loc[:,'outcome']    
    if continuous==True:
        data_in.loc[data_in['outcome'] >= data_in.outcome.mean(), 'outcomeD'] = 1
        data_in.loc[data_in['outcome'] <  data_in.outcome.mean(), 'outcomeD'] = 0
    
    outcomeD=data_in.loc[:,'outcomeD'].values

    lr_log_loss = np.nan
    roc_auc = np.nan
    brier = np.nan
    binom_p = np.nan
    Jeffreys_p =  np.nan
    
    max_outcome_fit=np.maximum(max(outcome), max(fit))
    min_outcome_fit=np.minimum(min(outcome), min(fit)) 
    if min_outcome_fit>=0 and max_outcome_fit<=1:
        lr_log_loss = log_loss(outcomeD, fit).round(4)
        roc_auc = roc_auc_score(outcomeD, fit).round(4)
        binom_p = binom_test(sum(outcomeD), n=len(outcomeD), p= np.mean(fit), alternative='greater').round(decimals=4)
        Jeffreys_p =  beta.cdf(np.mean(fit), sum(outcomeD)+0.5, len(outcomeD)-sum(outcomeD)+0.5).round(decimals=4)

            
    corr,_=pearsonr(fit,outcome)
    r2_OLS=corr**2
    
    the_table = [['Counts', len(outcome)],
                      ['Mean outcome', (sum(outcome)/len(outcome)).round(4)],
                      ['Mean fit', np.mean(fit).round(4)],
                      ['AUC ', roc_auc],
                      ['R-squared (OLS)', round(r2_OLS,4)],
                      ['R-squared', r2_score(outcome, fit).round(decimals=4)],
                      ['RMSE/ SQR(Brier score)', round(np.sqrt(((outcome-fit).dot(outcome-fit))/len(outcome)),4)],
                      ['Log loss', lr_log_loss], 
                      ['Binomial p-value', binom_p],
                      ['Jeffreys p-value', Jeffreys_p]]
    the_table=pd.DataFrame(data=the_table)
    the_table.columns = ['Metric', 'Value']
    
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
 
    plt.subplot(221)
    plt.title('Summary')
    plt.axis('off')
    plt.axis('tight')
    test=plt.table(cellText=the_table.values, colLabels=the_table.columns, loc='center', cellLoc='center', colWidths=[0.34, 0.2])
    test.auto_set_font_size(False)
    test.set_fontsize(16) 
    test.scale(2, 1.5)
    
    plt.subplot(222)
    plt.title('Time-Series Real-Fit')
    plt.plot(means['time'],means['outcome'])
    plt.plot(means['time'],means['fit'], color='red', ls='dashed')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Mean', fontsize=15)
    plt.tick_params(axis='both', labelsize=13)
    plt.legend(('Outcome','Fit'), loc='best', fontsize=15)
    
    plt.subplot(223)
    plt.title('Fit Histogram')
    plt.hist(fit, bins=20, histtype='bar', density=True)
    plt.xlabel('Fit', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.tick_params(axis='both', labelsize=13)
    
    data_in['cat'] = pd.qcut(data_in.fit, 10, labels=False, duplicates='drop')
    real_fit = data_in.groupby('cat')[['fit', 'outcome']].mean()
    mpv=real_fit.fit.values
    fop=real_fit.outcome.values
    
    maximum=np.maximum(max(fop), max(mpv))       
    maximum=np.ceil(maximum*100)/100
    minimum=np.minimum(min(fop), min(mpv))
    minimum=np.floor(minimum*100)/100
    
    plt.subplot(224)
    plt.title('Calibration Curve')
    plt.plot(mpv, fop, marker='.', linestyle='', markersize=18)
    plt.plot([minimum,maximum],[minimum,maximum], linestyle='--', color='gray')
    plt.xlim((minimum,maximum))
    plt.ylim((minimum,maximum))
    plt.xlabel('Mean fit', fontsize=15)
    plt.ylabel('Mean outcome', fontsize=15)
    plt.tick_params(axis='both', labelsize=13)
    plt.show()    
  
    
# Function resolutionbias

def resolutionbias(data_in, lgd, res, t):
        
    df = data_in
    
    df2=df.dropna(subset=[res]).copy()
    df2.loc[:,'res_period'] = df2.loc[:,res] - df2.loc[:,t]
    
    df2.loc[df2['res_period'] >= 20, 'res_period'] = 20

    data_LGD_sum = df2.groupby('res_period')[lgd].sum().reset_index(drop=False)
    data_LGD_count = df2.groupby('res_period')[lgd].count().reset_index(drop=False)
    
    data_LGD_sum = data_LGD_sum.sort_values(by='res_period', ascending=False)
    data_LGD_count = data_LGD_count.sort_values(by='res_period', ascending=False)

    data_LGD_sum_cumsum = data_LGD_sum.cumsum()
    data_LGD_count_cumsum = data_LGD_count.cumsum()
    data_LGD_mean = data_LGD_sum_cumsum/data_LGD_count_cumsum
    
    data_LGD_mean = data_LGD_mean.iloc[:,0:4]
    data_LGD_mean[t] = 61-data_LGD_mean.index
    data_LGD_mean = data_LGD_mean.set_index(t)

    data_LGD_mean2 = data_LGD_mean.iloc[np.full(41, 0)].reset_index(drop=True)
    data_LGD_mean3 = data_LGD_mean2.append(data_LGD_mean).reset_index(drop=False)
    data_LGD_mean3 = data_LGD_mean3.rename(columns={'index': t})

    df = df[df.loc[:,res].isnull()].drop([lgd], axis = 1) 
    df = pd.merge(df, data_LGD_mean3, on=t)   
    df3 = df2.append(df,sort=True)
    df3 = df3.drop('res_period', axis='columns').copy()
    return df3