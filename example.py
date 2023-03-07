from cost_sens_boost import IRLS
import pandas as pd
import statsmodels.api as sm
import lightgbm as lgb
import argparse
import logging

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default=10, type=int)
    parser.add_argument("--b", default=1, type=int)
    parser.add_argument("--type", default='linlin')
    return parser.parse_args()

if __name__ == '__main__':

    args = init_arg()
    df = pd.read_csv(r'house_KC_data.csv')
    df = sm.tools.tools.add_constant(df, prepend=True, has_constant='skip')
    features = ['intercept', 'bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade',
                'log_sqft_living', 'log_sqft_lot', 'log_sqft_above',
                'log_sqft_living15', 'log_sqft_lot15',
                'age', 'age_rnv']
    target_variable = "target_log"
    light_gbm = lgb.LGBMRegressor()
    light_gbm.fit(df[features], df[target_variable])
    pred_train = light_gbm.predict(df[features])
    res_train = pred_train - df[target_variable]
    beta,post_hoc_cost,initial_cost = IRLS(args.type,df[features],res_train,args.a,args.b)
    print("The below costs and One-Step Boosting algorithm is run using cost function",args.type, "with costs:"
          , args.a,"/",args.b)
    print("Initial average misprediction cost: ", initial_cost)
    print("Coefficients of boosting step: ", beta)
    print("Post-hoc average misprediction cost: ", post_hoc_cost)

