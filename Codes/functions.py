# Version 1.5

import os
import re
import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.metrics import mean_absolute_error as MAE

from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen, select_coint_rank

import warnings
warnings.filterwarnings('ignore')

# Statistic Tests
def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.") 
        
def kpss_test(x, name='', h0_type='c'):
    '''
    Perform KPSS test
    '''
    print(f'    KPSS Test on "{name}"', "\n   ", '-'*47)
    indices = ['Test Statistic', 'p-value', '# of Lags']
    kpss_test = kpss(x, regression=h0_type, nlags ='auto')
    results = pd.Series(kpss_test[0:3], index=indices)
    for key, value in kpss_test[3].items():
        results[f'Critical Value ({key})'] = value
        return results
    
def causation_matrix(data, variables, max_lag, test='ssr_chi2test', verbose=False):
    '''
    Perform Granger Causality Test
    '''
    X = pd.DataFrame(np.zeros((len(variables), len(variables)))
                     , columns=variables, index=variables)
    for c in X.columns:
        for r in X.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag = max_lag, verbose = False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(max_lag)]
            if verbose: 
                print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            X.loc[r, c] = min_p_value
    X.columns = [var + '-x axis' for var in variables]
    X.index = [var + '-y axis' for var in variables] 
      
    return X

def co_integration_test(endog, det_order, k_ar_diff):
    ''' 
    Co-integration Test 
    '''
    pd.options.display.float_format = "{:.2f}".format
    coint_test = coint_johansen(endog=endog, det_order=det_order, k_ar_diff=k_ar_diff)
    
    meig_df = pd.DataFrame(coint_test.cvm)
    meig_df.rename(columns={0:'90%', 1:'95%', 2:'99%'} ,inplace=True)
    trac_df = pd.DataFrame(coint_test.cvt)
    trac_df.rename(columns={0:'90%', 1:'95%', 2:'99%'} ,inplace=True)
    
    meig_stat = pd.DataFrame(coint_test.lr2).rename(columns={0:'Eigen-value Statistic'})
    trac_stat = pd.DataFrame(coint_test.lr1).rename(columns={0:'Trace Statistic'})
    
    print('Eigen-value Statistic')
    print(meig_stat)
    print('\n')
    print('Eigen-value Critical Values')
    print(meig_df)
    print('-----------------------------------------------')
    print('Trace Statistic')
    print(trac_stat)
    print('\n')
    print('Trace Critical Values')
    print(trac_df)

def fill_null_date(df, col):
    '''
    df: the dataframe to perform resampling
    col: the name of date column
    '''
    df_ind = df.set_index(pd.to_datetime(df[col])).drop(col, axis=1)
    df_ind = df_ind.resample("D").mean().fillna(np.nan)
    return df_ind

def hard_filter(df, varnames, hard_cut=4, fltr='hard', fill=True):
    '''
    This is a hard filter used for ATP and UW volume
    '''
    dfc = df.copy()
    
    for varname in varnames:
        dfc.loc[dfc[varname] <= hard_cut, varname] = np.nan        
        if fill:
            if fltr == 'hard':                
                dfc[varname] = dfc[varname].fillna(hard_cut)
            else:
                dfc[varname] = dfc[varname].interpolate(method='linear') 
            
    return dfc

# use date to index the dataframe and resample the data to fill null dates
def median_filter(df, varnames, window=24, std=3, fill=True):
    '''
    ----------
    df : pandas.DataFrame
        The pandas.DataFrame containing the column to filter.
    varname : string
        Column to filter in the pandas.DataFrame. No default. 
    window : integer 
        Size of the window around each observation for the calculation 
        of the median and std. Default is 24 (time-steps).
    std : integer 
        Threshold for the number of std around the median to replace 
        by `np.nan`. Default is 3 (greater / less or equal).
    Returns
    ----------
    dfc : pandas.Dataframe
        A copy of the pandas.DataFrame `df` with the new, filtered column `varname`
    '''
    dfc = df[varnames]
    
    for varname in varnames:
        
        dfc[varname + '_median']= dfc[varname].rolling(window, center=True).median()
        dfc[varname + '_std'] = dfc[varname].rolling(window, center=True).std()

        dfc.loc[dfc[varname] >= dfc[varname + '_median'] + std*dfc[varname + '_std'], varname] = np.nan
        dfc.loc[dfc[varname] <= dfc[varname + '_median'] - std*dfc[varname + '_std'], varname] = np.nan   
          
        if fill:
            dfc[varname] = dfc[varname].interpolate(method='linear')

    return dfc[varnames]

def draw_line_plot(col_list, df):
    '''
    Draw a line plot for the first glance at data
    '''
    n_figs = len(col_list)
    for i in range(0,n_figs):
        sub_df = df[col_list[i]]
        if 'date' in list(df.columns):
            x = df['date']
        else:
            x = pd.Series(list(df.index))
        plt.subplot(n_figs, 1, i+1)
        plt.plot(x, sub_df, '-', label=col_list[i])
        plt.legend(loc='upper left')
    plt.tight_layout()
    
def draw_scatter_plot(col_list, y_col, df, layout=(2,2)):
    '''
    Draw a line plot for the first glance at data
    layout[0]*layout[1] should be equal to len(col_list)
    '''
    n_figs = len(col_list)
    y = df[y_col]   
    
    for i in range(0,n_figs):
        x = df[col_list[i]]
        plt.subplot(layout[0], layout[1], i+1)
        plt.scatter(x, y, marker='.', color='b')
        plt.xlabel(col_list[i])
        plt.ylabel(y_col)
    plt.tight_layout()    
    
def get_MA(df, k, cols=[], dropna=True):
    '''
    Get a Moving Average result of a series
    '''
    df_bk = df.copy()
    if type(df) == pd.DataFrame:
        if cols:
            for col in cols:
                df_bk[col+'_MA'+str(k)] = df_bk[col].rolling(k).mean()
        else:
            print('A list of columns should be passed to parameter -cols-.')
    elif type(df) == pd.Series:
        df_bk = df_bk.rolling(k).mean()
    
    if dropna:
        df_bk.dropna(inplace=True)
        
    return df_bk

def get_train_df(df, ds_col, y_col, train_cut, vali_cut=None):
    '''
    reset_index_inplace is not allowed in series
    '''
    train_df = df[:train_cut]  
    train_df.drop([pd.to_datetime(train_cut)], axis=0, inplace=True)
    train_df = train_df.reset_index(drop=False) 
    train_df['ds'] = train_df[ds_col]
    train_df['y'] = train_df[y_col]   
    train_df.drop([ds_col], axis=1, inplace=True)
    train_df.drop([y_col], axis=1, inplace=True)
    
    if vali_cut:
        vali_df = df[train_cut:vali_cut]  
        vali_df.drop([pd.to_datetime(vali_cut)], axis=0, inplace=True)
        vali_df = vali_df.reset_index(drop=False) 
        vali_df['ds'] = vali_df[ds_col]
        vali_df['y'] = vali_df[y_col]  
        vali_df.drop([ds_col], axis=1, inplace=True)
        vali_df.drop([y_col], axis=1, inplace=True)
        
        test_df = df[vali_cut:] 
        # test_df.drop([pd.to_datetime(vali_cut)], axis=0, inplace=True)
    else:
        test_df = df[train_cut:]   
        # test_df.drop([pd.to_datetime(train_cut)], axis=0, inplace=True)
        
    test_df = test_df.reset_index(drop=False) 
    test_df['ds'] = test_df[ds_col]
    test_df['y'] = test_df[y_col]
    test_df.drop([ds_col], axis=1, inplace=True)
    test_df.drop([y_col], axis=1, inplace=True)
    
    print('Training sample size: ' + str(len(train_df)) + '.')
    try:
        print('Validation sample size: ' + str(len(vali_df)) + '.')
    except:
        print('No Validation Set.')
    print('Testing sample size: ' + str(len(test_df)) + '.')
    
    if vali_cut:
        return train_df, vali_df, test_df
    else:
        return train_df, test_df
    
def build_Lagged_Features(s, lag_list, col_name=None, dropna=True):
    '''
    Create lagged features.
    Parameters:
    ------
    s:        pandas.DataFrame or Series 
              the variables that needs to be lagged
    lag_list: list
              a list of desired lags
    col_name: string
              the column name for input sereis
    dropna:   bool
              whether to drop na values, default is True
    
    Returns:
    ------
    res: pandas.DataFrame 
    '''
    if type(s) is pd.DataFrame:
        new_dict={}
        for col_name in s:
            new_dict[col_name]=s[col_name]
            # create lagged Series
            for l in lag_list:
                new_dict['%s_lag%d' %(col_name,l)]=s[col_name].shift(l)
        # retrive the index of original df
        res=pd.DataFrame(new_dict,index=s.index)
    elif type(s) is pd.Series:
        # the_range=range(lag+1)
        res=pd.concat([s.shift(i) for i in lag_list],axis=1)
        if col_name:
            res.columns=[col_name + 'lag_%d' %i for i in lag_list]
        else:
            res.columns=['lag_%d' %i for i in lag_list]
    else:
        print('Only works for DataFrame or Series')
        return None    
    if dropna:
        return res.dropna()  
    else:
        return res 
    
def draw_heat_map(df, figure_size=(20,12), file_name='heatmap.png'):
    '''
    Draw a heat map for correlation matrix
    Parameter:
    -------
    df: the target dataframe to calculate heatmap
    figure_size: the figure size of the heatmap
    file_name: the file name to be saved
    
    Returns:
    -------
    '''
    cor_mat = df.corr()
    sns.set(rc = {'figure.figsize':figure_size})
    sns.heatmap(cor_mat, annot=True)
    plt.savefig(file_name)
    plt.show()
    
def make_verif(forecast, orig_data, y_col): 
    """
    Put together the forecast (coming from fbprophet) 
    and the overved data, and set the index to be a proper datetime index, 
    for plotting
    Parameters
    ----------
    forecast : pandas.DataFrame 
        The pandas.DataFrame coming from the `forecast` method of a fbprophet 
        model. 
    
    Returns
    -------
    forecast : 
        The forecast DataFrane including the original observed data.
    """
    verif = forecast.copy()
    
    verif.index = pd.to_datetime(verif.ds)
    verif.drop(['ds'], axis=1, inplace=True)
    
    if type(orig_data) == pd.Series:
        verif = verif.merge(orig_data, how='left', left_index=True, right_index=True)
    else:
        verif = verif.merge(orig_data[y_col], how='left', left_index=True, right_index=True)
        
    verif['y'] = verif[y_col]
    
    return verif

def plot_verif(verif, vali_cut, test_cut=None):
    """
    plots the forecasts and observed data. 
    Parameters
    ----------
    verif : pandas.DataFrame
        The `verif` DataFrame coming from the `make_verif` function in this package
    train_cut : string
        The date used to separate the training and test set. 
    Returns
    -------
    fig : matplotlib Figure object
    """
    
    # fig, ax = plt.subplots(figsize=(14, 8),dpi=300)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    train = verif.loc[:vali_cut,:]
    
    ax.plot(train.index, train.y, 'ko', markersize=3)
    
    ax.plot(train.index, train.yhat, color='steelblue', lw=0.5)
    
    ax.fill_between(train.index, train.yhat_lower, train.yhat_upper, color='steelblue', alpha=0.3)
    
    if test_cut:
        vali = verif.loc[vali_cut:test_cut,:]
    
        ax.plot(vali.index, vali.y, 'ro', markersize=3)

        ax.plot(vali.index, vali.yhat, color='coral', lw=0.5)

        ax.fill_between(vali.index, vali.yhat_lower, vali.yhat_upper, color='coral', alpha=0.3)

        ax.axvline(datetime.datetime.strptime(vali_cut, '%Y-%m-%d'), color='0.8', alpha=0.7)
        
        test = verif.loc[test_cut:,:]
    
        ax.plot(test.index, test.y, 'go', markersize=3)

        ax.plot(test.index, test.yhat, color='g', lw=0.5)

        ax.fill_between(test.index, test.yhat_lower, test.yhat_upper, color='g', alpha=0.3)
        
        ax.axvline(datetime.datetime.strptime(test_cut, '%Y-%m-%d'), color='0.8', alpha=0.7)
    else:
        test = verif.loc[vali_cut:,:]
    
        ax.plot(test.index, test.y, 'ro', markersize=3)

        ax.plot(test.index, test.yhat, color='coral', lw=0.5)

        ax.fill_between(test.index, test.yhat_lower, test.yhat_upper, color='coral', alpha=0.3)

        ax.axvline(datetime.datetime.strptime(vali_cut, '%Y-%m-%d'), color='0.8', alpha=0.7)

    ax.grid(ls=':', lw=0.5)
    
    return fig

import matplotlib.ticker as mticker

def plot_test(test_df):
    '''
    Plot the results on testing samples
    '''
    # fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(test_df.index, test_df.y, 'ro', markersize=3)
    
    ax.plot(test_df.index, test_df.yhat, color='coral', lw=0.5)
    
    ax.fill_between(test_df.index, test_df.yhat_lower, test_df.yhat_upper, color='coral', alpha=0.3) 
    
    tick_spacing = test_df.index.size/5
    ax.xaxis.set_major_locator(mticker.MultipleLocator(tick_spacing))
    
    mae = MAE(test_df['y'].values, test_df['yhat'].values)
    cor = np.corrcoef(test_df['y'].values, test_df['yhat'].values)[0,1]
    
    print('MAE on Testing Sample: ' + str(mae))
    print('Correlation on Testing Sample: ' + str(cor))
    
    return fig

def plot_joint_plot(verif, x_scale, y_scale, legnd, x='yhat', y='y', title=None, fpath = '../Tushar_Volume Forecast', fname = None): 
    """   
    Parameters
    ---------- 
    verif : pandas.DataFrame, apply cut-off when plot training/testing set 
    x : string 
        The variable on the x-axis
        Defaults to `yhat`, i.e. the forecast or estimated values.
    y : string 
        The variable on the y-axis
        Defaults to `y`, i.e. the observed values
    title : string 
        The title of the figure, default `None`. 
    
    fpath : string 
        The path to save the figures, default to `../figures/paper`
    fname : string
        The filename for the figure to be saved
        ommits the extension, the figure is saved in png, jpeg and pdf
 
    Returns
    -------
    f : matplotlib Figure object
    """

    g = sns.jointplot(x='yhat', y='y', data = verif, kind="reg", color="0.4")
    
    g.fig.set_figwidth(8)
    g.fig.set_figheight(8)

    ax = g.fig.axes[1]
    
    if title is not None: 
        ax.set_title(title, fontsize=16)

    ax = g.fig.axes[0]

    ax.set_xlim(x_scale)
    ax.set_ylim(y_scale)

    ax.text(legnd[0], legnd[1]
            , "R = {:+4.2f}\nMAE = {:4.1f}".format(verif.loc[:,['y','yhat']].corr().iloc[0,1], 
                                                            MAE(verif.loc[:,'y'].values, verif.loc[:,'yhat'].values))
            , fontsize=16)

    ax.set_xlabel("model's estimates", fontsize=15)
    
    ax.set_ylabel("observations", fontsize=15)
    
    ax.grid(ls=':')

    [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
    [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()];

    ax.grid(ls=':')
    
    if fname is not None: 
        for ext in ['png']: 
            # ,'jpeg','pdf'
            g.fig.savefig(os.path.join(fpath, "{}.{}".format(fname, ext)), dpi=200)

def univar_model_train(train_df, test_df, y_mode, y_scale, cps, hps, w_mode, w_scale, f_period=None, **kwargs):
    """
    Train the uni-variate model with Pophet. 
    Parameters
    ----------
    train_df : pandas.DataFrame
               the dataframe that contains training set.
    test_df :  pandas.DataFrame 
               the dataframe that contains testing set
    y_mode :   string 
               "additive" or "multiplicative"
    y_scale :  float
               the prior scale of year seasonality
    cps :      float 
               the prior scale of change points
    hps :      float 
               the prior scale of holiday seasonality     
    w_mode :   string 
               "additive" or "multiplicative"
    w_scale :  float
               the prior scale of week seasonality               
    f_period ：int 
               the term to make predictions
    some other inputs stored in kwargs dictionary.
    Returns
    -------
    model :     an instance of prophet model class
    future :    future dates that predictions are made on
    forecast  : forecasted results
    """
   
    # Silencing the log information from cmdstanpy
    # For more reference, please check: https://mc-stan.org/cmdstanpy/_modules/cmdstanpy/utils.html
    import logging
    logger = logging.getLogger('cmdstanpy')
    logger.setLevel(logging.ERROR)
    handler = logging.StreamHandler()
    handler.setLevel(logging.ERROR)

    model = Prophet(yearly_seasonality=10
                    , seasonality_mode = y_mode
                    , seasonality_prior_scale=y_scale                    
                    , changepoint_prior_scale=cps
                    , changepoint_range = 1
                    , daily_seasonality=False
                    , weekly_seasonality=False
                    , holidays_prior_scale=hps)
    model.add_country_holidays(country_name='AU') 
    
    model.add_seasonality(name='weekly', period=7, fourier_order=3, mode=w_mode, prior_scale=w_scale)
    
    if 'quater' in kwargs.keys():
        if kwargs['quater']:
            model.add_seasonality(name='quaterly', period=91.25, fourier_order=7, mode=kwargs['q_mode'], prior_scale=kwargs['q_scale'])
 
    if 'month' in kwargs.keys():
        if kwargs['month']:
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5, mode=kwargs['m_mode'], prior_scale=kwargs['m_scale'])
           
    model.fit(train_df)     
    if f_period:
        future = model.make_future_dataframe(periods=f_period, freq='1D')
    else:
        future = model.make_future_dataframe(periods=len(test_df), freq='1D')
    forecast = model.predict(future)
    
    return model, future, forecast  


def unimodel_grid_search(train_df, test_df, y_mode_list, y_scale_list, cps_list, hps_list, 
                         w_mode_list, w_scale_list, q_ind, q_mode_list, q_scale_list, 
                         m_ind, m_mode_list, m_scale_list, train_cut, mae_w=0.5, cor_w=0.5):
    """
    Run the grid seach process for Uni-variate model. 
    Parameters
    ----------
    train_df :      pandas.DataFrame
                    the dataframe that contains training set.
    test_df :       pandas.DataFrame 
                    the dataframe that contains testing set
    y_mode_list :   list
                    the list for modes of yearly seasonality
    y_scale_list :  list 
                    the list for prior scales of year seasonality
    cps_list :      list 
                    the list for prior scales of change points
    hps_list :      list  
                    the list for prior scales of holiday seasonality     
    w_mode_list :   list 
                    the list for modes of weekly seasonality
    w_scale_list :  list 
                    the list for prior scales of week seasonality               
    q_ind ：        list  
                    list of bools. True or False(With/Without Quaterly Seasonality)
    q_mode_list ：  list 
                    the list for modes of quater seasonality
    q_scale_list ： list 
                    the list for prior scales of quater seasonality           
    m_ind ：        list  
                    list of bools. True or False(With/Without Monthly Seasonality)
    m_mode_list ：  list  
                    the list for modes of monthly seasonality
    m_scale_list ： list 
                    the list for prior scales of monthly seasonality   
    some other inputs stored in kwargs dictionary.
    Returns
    -------
    model_list :     all models during grid search
    forecast_list  : all forecast results during grid search
    mae_list :       the list of MAE metric
    cor_list :       the list of CORRELATION metric
    loss_list :      the list of losses during grid search(a combination of mae and cor)
    paras :          the list of all candidate paramters during grid search
    """    
    model_list = []
    forecast_list = []
    mae_list = []
    cor_list = []
    loss_list = []
    
    paras = list(itertools.product(y_mode_list, y_scale_list
                                   , cps_list, hps_list
                                   , w_mode_list, w_scale_list
                                   , q_ind, q_mode_list, q_scale_list
                                   , m_ind, m_mode_list, m_scale_list))
    for para in paras:
        uni_model, future, forecast = univar_model_train(train_df=train_df, test_df=test_df
                                                         , y_mode=para[0], y_scale=para[1]
                                                         , cps=para[2], hps=para[3] 
                                                         , w_mode=para[4], w_scale=para[5]
                                                         , f_period=None
                                                         , quater=para[6], q_mode=para[7], q_scale=para[8]
                                                         , month=para[9], m_mode=para[10], m_scale=para[11])
        
        model_list.append(uni_model)
        forecast_list.append(forecast)
        forecast.index = pd.to_datetime(forecast.ds)
        forecast.drop(['ds'], axis=1, inplace=True)
        mae = MAE(test_df['y'].values, forecast.loc[train_cut:,'yhat'].values)
        cor = np.corrcoef(test_df['y'].values, forecast.loc[train_cut:,'yhat'].values)[0,1]
        # test_df['y'].corr(forecast.loc[train_cut:,'yhat'])
        mae_list.append(mae)
        cor_list.append(cor)
        loss_list.append(mae_w*mae - cor_w*50*cor)
    
    min_loss = min(loss_list)
    min_pos = loss_list.index(min_loss)
    print('Minimum Position: ' + str(min_pos))
    print('Minimum Loss: ' + str(min_loss))
    print('MAE: ' + str(mae_list[min_pos]))
    print('Correlation: ' + str(cor_list[min_pos]))
    print('Best Parameters: ')
    print('y_mode: ', 'y_scale: ', 'cps: ', 'hps: ', 'w_mode: ', 'w_scale: ', 'q_ind: ', 'q_mode: ', 'q_scale: ', 'm_ind: ', 'm_mode: ', 'm_scale: ')
    print(paras[min_pos])
    
    return model_list, forecast_list, mae_list, cor_list, loss_list, paras

def add_regressor_to_future(future, regressors):
    '''
    Add extra regressor to make future forecast. 
    (ONLY USED FOR MODEL WITH EXTRA REGRESSOR)  
    '''
    futures = future.copy()
    futures.index = pd.to_datetime(futures.ds)
    
    if type(regressors) == list:
        if len(regressors) > 1:
            regressors_df = pd.concat(regressors, axis=1)
        else:
            regressors_df = regressors[0]
    elif type(regressors) == pd.DataFrame or type(regressors) == pd.Series:
        regressors_df = regressors
        
    futures = futures.merge(regressors_df, how='left', left_index=True, right_index=True)
    futures = futures.reset_index(drop = True)
    
    return futures  

def add_regressors(data, regressors):
    '''
    Add extra regressor during model training. 
    (ONLY USED FOR MODEL WITH EXTRA REGRESSOR)  
    '''   
    data_with_regressors = data.copy()   
    data_with_regressors.index = pd.to_datetime(data_with_regressors.ds)
    # data_with_regressors.drop(['ds'], axis=1, inplace=True)
    
    if type(regressors) == list:
        if len(regressors) > 1:
            regressors_df = pd.concat(regressors, axis=1)
        else:
            regressors_df = regressors[0]
    elif type(regressors) == pd.DataFrame or type(regressors) == pd.Series:
        regressors_df = regressors
    data_with_regressors = data_with_regressors.merge(regressors_df, how='left'
                                                      , left_index=True, right_index=True)
    data_with_regressors = data_with_regressors.reset_index(drop = True)
    
    return data_with_regressors

def get_lagged_extra_regressor(regressor, lag, dcol, vcol):
    regressor = regressor.reset_index()
    regressor[dcol] = regressor[dcol] + timedelta(days=lag)
    regressor.index = pd.to_datetime(regressor[dcol])
    regressor = regressor[vcol]
    return regressor

def get_combines(collect, sub_num, mode='Single'):
    if mode == 'Single':
        combines = list(itertools.combinations(collect, sub_num))
    elif mode == 'Multi':
        extend_collect = []
        for i in range(0, sub_num):
            extend_collect.extend(collect)
        combines = list(set(list(itertools.permutations(extend_collect, sub_num))))
    else:
        print('Pass wrong mode.')
        combines = []       
    return combines

def multivar_model_train(train_df, test_df, extra_regs, y_mode, y_scale, cps, hps, w_mode, w_scale,
                         reg_names, reg_modes, stands, reg_prior_scales, lag=0, f_period=None, **kwargs):
    """
    Train the multi-variate model with Pophet. 
    Parameters
    ----------
    train_df : pandas.DataFrame
               the dataframe that contains training set.
    test_df :  pandas.DataFrame 
               the dataframe that contains testing set
    y_mode :   string 
               "additive" or "multiplicative"
    y_scale :  float
               the prior scale of year seasonality
    cps :      float 
               the prior scale of change points
    hps :      float 
               the prior scale of holiday seasonality     
    w_mode :   string 
               "additive" or "multiplicative"
    w_scale :  float
               the prior scale of week seasonality   
    reg_names: list
               the list for names of extra regressors
    reg_modes: list
               "additive" or "multiplicative"
    stands:    list
               Standardize the extra regressor, True or False
    reg_prior_scales: list
               The list for prior scales of extra regressors
    lag:       int 
               The number of lags taken on the extra regressors
    f_period ：int 
               the term to make predictions
    some other inputs stored in kwargs dictionary.
    Returns
    -------
    model :     an instance of prophet model class
    futures :   future dates that predictions are made on
    forecast  : forecasted results
    """
    
    # Silencing the log information from cmdstanpy
    # For more reference, please check: https://mc-stan.org/cmdstanpy/_modules/cmdstanpy/utils.html
    import logging
    logger = logging.getLogger('cmdstanpy')
    logger.setLevel(logging.ERROR)
    handler = logging.StreamHandler()
    handler.setLevel(logging.ERROR)
    
    model = Prophet(yearly_seasonality=10
                    , seasonality_mode = y_mode                   
                    , seasonality_prior_scale=y_scale
                    , changepoint_prior_scale=cps
                    , weekly_seasonality=False
                    , daily_seasonality=False
                    , holidays_prior_scale=hps)
    
    model.add_country_holidays(country_name='US')

    model.add_seasonality(name='weekly', period=7, fourier_order=3, mode=w_mode, prior_scale=w_scale)
    
    if 'quater' in kwargs.keys():
        if kwargs['quater']:
            model.add_seasonality(name='quaterly', period=91.25, fourier_order=7
                                  , mode=kwargs['q_mode'], prior_scale=kwargs['q_scale'])
 
    if 'month' in kwargs.keys():
        if kwargs['month']:
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5
                                  , mode=kwargs['m_mode'], prior_scale=kwargs['m_scale'])
    
    if type(reg_names) == tuple:
        for i in range(0, len(reg_names)):
            model.add_regressor(reg_names[i], mode=reg_modes[i], standardize=stands[i], prior_scale=reg_prior_scales[i])
    else:
        model.add_regressor(reg_names, mode=reg_modes, standardize=stands, prior_scale=reg_prior_scales)
        
    train_df_with_regs = add_regressors(train_df, extra_regs)
    model.fit(train_df_with_regs)
    
    if f_period:
        if len(test_df) + lag < f_period:
            print('The number of forecast periods should be smaller than the sum of test sample size and lag.')
            print('Fall back to test sample size.')
            future = model.make_future_dataframe(periods=len(test_df), freq='1D')
        else:
            future = model.make_future_dataframe(periods=f_period, freq='1D')
    else:
        future = model.make_future_dataframe(periods=len(test_df), freq='1D')
        
    futures = add_regressor_to_future(future, extra_regs)   
    forecast = model.predict(futures)
    
    return model, futures, forecast

def multivar_grid_search(train_df, test_df, extra_regs, y_mode_list, y_scale_list,
                         cps_list, hps_list, w_mode_list, w_scale_list,
                         reg_names_list, reg_modes_list, stands_list, reg_prior_scales_list, 
                         q_ind, q_mode_list, q_scale_list,m_ind, m_mode_list, m_scale_list, train_cut, mae_w, cor_w):
    """
    Run the grid seach process for multi-variate model. 
    Parameters
    ----------
    train_df :      pandas.DataFrame
                    the dataframe that contains training set.
    test_df :       pandas.DataFrame 
                    the dataframe that contains testing set
    y_mode_list :   list
                    the list for modes of yearly seasonality
    y_scale_list :  list 
                    the list for prior scales of year seasonality
    cps_list :      list 
                    the list for prior scales of change points
    hps_list :      list  
                    the list for prior scales of holiday seasonality     
    w_mode_list :   list 
                    the list for modes of weekly seasonality
    w_scale_list :  list 
                    the list for prior scales of week seasonality               
    q_ind ：        list  
                    list of bools. True or False(With/Without Quaterly Seasonality)
    q_mode_list ：  list 
                    the list for modes of quater seasonality
    q_scale_list ： list 
                    the list for prior scales of quater seasonality           
    m_ind ：        list  
                    list of bools. True or False(With/Without Monthly Seasonality)
    m_mode_list ：  list  
                    the list for modes of monthly seasonality
    m_scale_list ： list 
                    the list for prior scales of monthly seasonality   
    some other inputs stored in kwargs dictionary.
    Returns
    -------
    model_list :     all models during grid search
    forecast_list  : all forecast results during grid search
    mae_list :       the list of MAE metric
    cor_list :       the list of CORRELATION metric
    loss_list :      the list of losses during grid search(a combination of mae and cor)
    paras :          the list of all candidate paramters during grid search
    """       
    model_list = []
    forecast_list = []
    mae_list = []
    cor_list = []
    loss_list = []
    
    # verif.loc[:,['y','yhat']].corr().iloc[0,1]
    paras = list(itertools.product(y_mode_list, y_scale_list,
                                   cps_list, hps_list, 
                                   w_mode_list, w_scale_list,
                                   reg_names_list, reg_modes_list, stands_list, reg_prior_scales_list, 
                                   q_ind, q_mode_list, q_scale_list, 
                                   m_ind, m_mode_list, m_scale_list)) 
    for para in paras:
        multivar_model, multivar_futures, multivar_forecast = multivar_model_train(train_df, test_df, extra_regs, 
                                                                                   para[0], para[1], 
                                                                                   para[2], para[3],
                                                                                   para[4], para[5],
                                                                                   para[6], para[7], para[8], para[9],
                                                                                   lag=0, f_period=None,
                                                                                   quater=para[10], q_mode=para[11], q_scale=para[12],
                                                                                   month=para[13], m_mode=para[14], m_scale=para[15])
   
        model_list.append(multivar_model)
        forecast_list.append(multivar_forecast)
        multivar_forecast.index = pd.to_datetime(multivar_forecast.ds)
        multivar_forecast.drop(['ds'], axis=1, inplace=True)
        
        mae = MAE(test_df['y'].values, multivar_forecast.loc[train_cut:,'yhat'].values)
        cor = np.corrcoef(test_df['y'].values, multivar_forecast.loc[train_cut:,'yhat'].values)[0,1]
        # test_df['y'].corr(forecast.loc[train_cut:,'yhat'])
        mae_list.append(mae)
        cor_list.append(cor)
        loss_list.append(mae_w*mae - cor_w*50*cor)        
    
    min_loss = min(loss_list)
    min_pos = loss_list.index(min_loss)
    print('Minimum Position: ' + str(min_pos))
    print('Minimum Loss: ' + str(min_loss))
    print('MAE: ' + str(mae_list[min_pos]))
    print('Correlation: ' + str(cor_list[min_pos]))
    print('Best Parameters: ')
    print('y_mode: ', 'y_scale: ', 
          'cps: ', 'hps: ', 
          'w_mode: ', 'w_scale: ', 
          'reg_name: ', 'reg_mode: ', 'reg_stand: ', 'rps: ', 
          'q_ind: ', 'q_mode: ', 'q_scale: ', 
          'm_ind: ', 'm_mode: ', 'm_scale: ')
    print(paras[min_pos])    
    
    return model_list, forecast_list, mae_list, cor_list, loss_list, paras  
