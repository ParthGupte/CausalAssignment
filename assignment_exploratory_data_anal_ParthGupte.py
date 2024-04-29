import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
from scipy.special import logit
import scipy.stats as stats
import numpy as np

nsw_df = pd.read_csv("nsw.csv")
cps_df = pd.read_csv("cps.csv")

sig_cutoff = 0.05

def compare_cts(col_name:str,df:pd.DataFrame = nsw_df):
    treatment_grp = df[df["treat"] == 1][col_name]
    control_grp = df[nsw_df["treat"] == 0][col_name]
    plt.hist(control_grp,color="r",label="Control Group",weights=np.ones(len(control_grp))/len(control_grp))
    plt.hist(treatment_grp,color="g",label="Treatment Group",alpha=0.7,weights=np.ones(len(treatment_grp))/len(treatment_grp))
    plt.legend()
    plt.title(col_name+" Treatment vs Control")
    plt.savefig(col_name+"_cts_hist_plot")
    plt.show()
    plt.close()
    t = stats.ttest_ind(treatment_grp,control_grp)
    return t.statistic, t.pvalue

def compare_bin(col_name:str,df:pd.DataFrame = nsw_df):
    treatment_grp = df[df["treat"] == 1][col_name]
    control_grp = df[df["treat"] == 0][col_name]
    plt.hist(control_grp,color="r",label="Control Group",weights=np.ones(len(control_grp))/len(control_grp))
    plt.hist(treatment_grp,color="g",label="Treatment Group",alpha=0.7,weights=np.ones(len(treatment_grp))/len(treatment_grp))
    plt.legend()
    plt.title(col_name+" Treatment vs Control")
    plt.savefig(col_name+"_bin_hist_plot")
    plt.show()
    plt.close()
    trt_pos = treatment_grp.sum()
    # print(treatment_grp.iloc[:10])
    ctrl_pos = control_grp.sum()
    trt_neg = treatment_grp.shape[0]-trt_pos
    ctrl_neg = control_grp.shape[0] - ctrl_pos
    cont_table = [[trt_neg,trt_pos],[ctrl_neg,ctrl_pos]]
    # print(treatment_grp.shape[0],cont_table)
    chi = stats.chi2_contingency(cont_table)
    return chi[0], chi[1]



cts_cols = ["age","re74","re75","age2","ed"]
bin_cols = ["black","hisp","married","nodeg"]

for col in cts_cols:
    t,p = compare_cts(col)
    print("Column {} has T-statistic value = {} with p-value = {}".format(col,t,p))
    if p > sig_cutoff:
        print("Hence we cannot say that the distributions are diffrent with significance (<{})".format(sig_cutoff))
    else:
        print("Hence the distributions are the same with significance (<{})".format(sig_cutoff))
    print()

for col in bin_cols:
    chi, p = compare_bin(col)
    print("Column {} as chi-sq value = {} with p-value = {}".format(col,chi,p))
    if p > sig_cutoff:
        print("Hence we cannot say that the distributions are diffrent with significance (<{})".format(sig_cutoff))
    else:
        print("Hence the distributions are the same with significance (<{})".format(sig_cutoff))
    print()