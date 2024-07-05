# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:15:54 2021

@author: z00119

2021/3/31 左轴刻度把样本数改成样本占比
"""


import pandas as pd
import numpy as np
from sklearn import metrics 
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import style
plt.style.use('seaborn-v0_8')
import pylab
pylab.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
pylab.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
import warnings
warnings.filterwarnings('ignore')


#%%
# 该函数使用于模型分夸月看效果
def plt_multi_rsk_trend(df,x,y='fpd_k4',miss_values=[-999],has_dt=1,dt ='event_date',dt_cut='month',score_cut=None,n=10,name = ''):
    '''
    Parameters
    ----------
    df : pd.dataframe
        输入dataframe,需包含y，dt.
    x : string
        模型分 如'ali_cs_duotou_score',连续值
    y : string
        样本的因变量 如 'fpd4', 因变量只能为0,1的int格式输入.
    miss_values: list
        缺失值列表，缺失值单独一箱
    has_dt: 0,1
        针对没有dt的dataframe
    dt : string, optional
        样本拆分的日期，一般我们建模中常用的，建案时间，交易时间等等。格式为 20190101、2019-01-01均可. The default is 'event_date'.
    n : int, optional
        特征等频分箱的组数，默认q=10，等频分为10箱. The default is 10.
    dt_cut : list, optional
        样本拆分月自定义切点，list自定义 如:[-np.inf,20190101,20190601,20190901,np.inf]
        有"month","dt" 和自定义切点三种模式
        "month"--- 按照dt拆分每月看
        "dt" --- 按照dt字段看-这里dt就可以自定义是周的字段还是其他
    score_cut : list, optional
        默认值 None的时候是按照n=xx等频切分，可自定义固定模型分切点
    name : string, optional
        图片名称. The default is ''.

    Returns
    -------
    plot图片.
    bins -- 模型分的切点，可用在别的样本分组上
    风险分布明细数据

    '''
    def cal_ks(data,prob,y):
        return ks_2samp(data[prob][data[y]==1],data[prob][data[y]==0]).statistic

    def cal_iv(x,y):
        df = y.groupby(x).agg(['count','sum'])
        df.columns = ['total','count_1']
        df['count_0'] = df.total - df.count_1
        df['pct'] = df.total/df.total.sum()
        df['bad_rate'] = df.count_1/df.total
        df['woe'] = np.log((df.count_0/df.count_0.sum())/(df.count_1/df.count_1.sum()))
        df['woe'] = df['woe'].replace(np.inf,0)
        #    print(df)
        rate = ((df.count_0/df.count_0.sum())-(df.count_1/df.count_1.sum()))
        IV = np.sum(rate*df.woe)
        return IV,df
    
    if has_dt==1:
        temp_df = df[[x,y,dt]]
        temp_df[x] = temp_df[x].replace(miss_values,-99)
        if dt_cut=='month':
            temp_df['dt_cut'] = temp_df[dt].astype('str').str.replace('-','').str[0:6]
        elif dt_cut =='dt':
            temp_df['dt_cut'] = temp_df[dt]
        else:
            temp_df[dt] = temp_df[dt].astype('str').str.replace('-','').astype(int)
            temp_df['dt_cut'] = pd.cut(temp_df[dt],dt_cut).astype('object')
            
    else:
        temp_df = df[[x,y]]
        temp_df[x] = temp_df[x].replace(miss_values,-99)
        temp_df['dt_cut'] = 'all'
    
    if score_cut!=None:
        temp_df['group'] = pd.cut(temp_df[x],score_cut)
        bins = score_cut
    else:
        _,bins = pd.qcut(temp_df[x],n,duplicates='drop',retbins=True)
        bins[0] = -99
        bins[-1] = np.inf
        bins = sorted(set(np.insert(bins,0,[-np.inf,-99,np.inf])))
        temp_df['group'] = pd.cut(temp_df[x],bins)

    a = temp_df.pivot_table(index=['dt_cut','group'],values=y,aggfunc=np.size,fill_value=0)
    a = a.unstack(level=0).droplevel(0,axis=1)
    a.index = a.index.astype('str')
    pct = a/a.sum()
    A = pd.concat([(pct.iloc[:,i]-pct.iloc[:,0])*np.log(pct.iloc[:,i]/pct.iloc[:,0]) for i in range(pct.shape[1])],axis=1)
    A.columns = pct.columns
    psi = A.sum()
    pct = pct.fillna(0)
    
    
    y_list = temp_df.groupby(temp_df['dt_cut'])[y].apply(lambda x:sum(x)/len(x))   
    num_list = temp_df.groupby(temp_df['dt_cut'])[y].apply(lambda x:len(x))   
    iv_list = temp_df.groupby(temp_df['dt_cut']).apply(lambda k: cal_iv(k['group'],k[y])[0])
    ks_list = temp_df.groupby(temp_df['dt_cut']).apply(lambda k1: cal_ks(k1,x,y))
    auc_list = temp_df.groupby(temp_df['dt_cut']).apply(lambda k2: metrics.roc_auc_score(k2[y],k2[x]))

    c = temp_df.pivot_table(index=['group'],values=[y],columns='dt_cut',aggfunc=[len,np.sum,lambda x:sum(x)/len(x)],fill_value=0)
    c.columns = c.columns.set_levels(['len','sum','ratio'], level=0)
    c.columns = c.columns.droplevel(1)

#    tot_y = temp_df[y].sum()/temp_df[y].shape[0] 
    tot_size = temp_df[y].shape[0] 
    tot_iv = cal_iv(temp_df['group'],temp_df[y])[0]
    tot_ks = cal_ks(temp_df,x,y)
    tot_auc = metrics.roc_auc_score(temp_df[y],temp_df[x])
        
    l = sorted(set(temp_df['dt_cut']))
    
        
    fig = plt.figure(figsize=(6*len(l),4))
    fig.suptitle('{}{},{}, 总样本量：{}, 总IV:{:.2f}, 总KS:{:.2f}, 总AUC:{:.2f}'.format(x,name,y,tot_size,tot_iv,tot_ks,tot_auc),x = 0.08, y = 1.07, ha='left',size=15,bbox = dict(facecolor = 'grey',alpha=0.1))
    
    for k in l:  
        i = l.index(k)    
        max_bs = c[('ratio',k)].iloc[-1]/y_list[k]
        
        ax1 = plt.subplot(1,len(l),i+1)
        ax1.set_xticklabels([str(x) for x in list(c.index)],rotation=90,fontsize=10)
        sns.barplot(x=list(c.index), y=pct[k],ax = ax1,alpha=0.2,color = 'k')
        ax1.text(0.01, 0.95, "平均风险:{:.2%}  {:.1f}倍".format(y_list[k],max_bs), transform=ax1.transAxes, fontdict={'size': '10', 'color': 'b'}) # 写平均风险值
        ax1.set_title('{}, 样本量：{}, \n IV:{:.2f}, KS:{:.2f}, AUC:{:.2f}, PSI:{:.2f}'.format(str(k).replace('.0',''),num_list[k],iv_list[k],ks_list[k],auc_list[k], psi[k]),size = 12) #表标题
        ax1.set_ylim([0,pct.max().max()*1])
        ax1.axes.set_ylabel('')
        
        ax2 = ax1.twinx()
        sns.pointplot(list(c.index), c[('ratio',k)], ax=ax2, alpha=0.2,color='red',scale=0.5)  # 画在ax2画布上
        for a,b in zip([i for i in range(len(c.index))], c[('ratio',k)]):
            ax2.annotate("{:.2%}".format(b), xy=(a,b), xytext=(-20, 5), textcoords='offset points', weight='heavy')
        ax2.axhline(y=y_list[k], color='grey', label='avg_risk', linestyle='--', linewidth=1) # 平均风险线
        ax2.set_ylim([0,c['ratio'].max().max()*1.2])
        ax2.axes.set_ylabel('')
    
    plt.show()
    plt.close()
    return bins,c
    # fig.savefig(file,bbox_inches='tight')

#%%
# 使用样例
# df = pd.read_csv('ts_y_11to01.csv')
# # df = df[['rsk_log_id', 'event_date','fpd_k4', 'mob3_k10', 'pre1', 'pre3', 'pre_lr']]

# #%%
# plt_multi_rsk_trend(df,x='dz_bcash_t3_xgb01_score',y='fpd4',miss_values=[-999],
#                     has_dt=1,dt ='month',dt_cut='dt',score_cut=None,n=10,name = '')
