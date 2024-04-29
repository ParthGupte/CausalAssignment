import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
from scipy.special import logit

nsw_df = pd.read_csv("nsw.csv")
cps_df = pd.read_csv("cps.csv")

#fitting linear regression
nsw_model = LinearRegression()
cps_model = LinearRegression()

nsw_model.fit(nsw_df[["treat","age","ed","black","hisp","married","nodeg","re74","re75","age2"]],nsw_df["re78"])
cps_model.fit(cps_df[["treat","age","ed","black","hisp","married","nodeg","re74","re75","age2"]],cps_df["re78"])

def get_named_coefs(model:LinearRegression):
    coef = model.coef_
    if coef.ndim > 1:
        coef = list(coef)[0]
    coef_dict = dict(zip(model.feature_names_in_,coef))
    coef_dict["intercept"] = float(model.intercept_)
    return coef_dict

def d_PS(ps1:float,ps2:float):
    return (logit(ps1)-logit(ps2))**2

def find_closest_pt(source_pt:float,target_series:pd.Series):
    D = {}
    for idx_C, ps_C in target_series.items():
        D[idx_C] = d_PS(source_pt,ps_C)
    clst_pt = sorted(list(D.items()),key=lambda x:x[1])[0][0]
    return clst_pt



coef_dict_nsw = get_named_coefs(nsw_model) 
coef_dict_cps = get_named_coefs(cps_model)
print("Model Coefs for NSW linear regression model:",coef_dict_nsw,sep="\n")
print()
print("Model coefs for CPS linear regression model:",coef_dict_cps,sep="\n")
print()

#fitting logistic regression
propensity_model_nsw = LogisticRegression()

propensity_model_nsw.fit(nsw_df[["age","ed","black","hisp","married","nodeg","re74","re75","age2"]],nsw_df["treat"])
coef_dict_propensity = get_named_coefs(propensity_model_nsw)
print("Model Coefs for propensity score model:",coef_dict_propensity,sep="\n")
print()
p = propensity_model_nsw.predict_proba(nsw_df[["age","ed","black","hisp","married","nodeg","re74","re75","age2"]])
prop_scores = [p_i[1] for p_i in p]
nsw_df.insert(11,"propensity_score", prop_scores)

#plot scores
treatment_grp_scores = nsw_df[nsw_df["treat"] == 1]["propensity_score"]
control_grp_scores = nsw_df[nsw_df["treat"] == 0]["propensity_score"]
plt.hist(control_grp_scores,color="r",label="Control Group")
plt.hist(treatment_grp_scores,color="g",label="Treatment Group",alpha = 0.7)
plt.legend()
# plt.show() #they have a common base

#matching
# we will simply find the idx in control group with least distance from given treatment point for matching (with replacement)
matches_dict = {} #{idx_treament:idx_control}
for idx_T, ps_T in treatment_grp_scores.items():
    matches_dict[idx_T] = find_closest_pt(ps_T,control_grp_scores)

print("Dictionary of matches: ",matches_dict)
print()
idxs = []
idxs.extend(list(matches_dict.keys()))
idxs.extend(list(matches_dict.values()))

new_nsw_df = nsw_df.filter(items=idxs)
print("Size of filtered database: ",len(new_nsw_df),"Size of orginal database: ",len(nsw_df))

new_nsw_model = LinearRegression()
new_nsw_model.fit(nsw_df[["treat","age","ed","black","hisp","married","nodeg","re74","re75","age2"]],nsw_df["re78"])
coef_dict_new_nsw = get_named_coefs(new_nsw_model)
print("Model Coefs for new NSW linear regression model:",coef_dict_new_nsw,sep="\n")
print()
print("Model Coefs for NSW linear regression model:",coef_dict_nsw,sep="\n")
print()



