
import pandas as pd
import numpy as np
from lightgbm.sklearn import LGBMClassifier 
from sklearn.ensemble import RandomForestClassifier

 
 
 

def get_data(data_pd):
    data = data_pd.iloc[:,3:]
     
    label = data_pd.iloc[:,2]
     
   
    return data,label

def LGBM_SF(X,y,fileName):#Light Gradient Boosting Machine Feature Selection

    f=fileName+"_"
    print("LGBoosting.....")
    model = LGBMClassifier(num_leaves=32,n_estimators=888,max_depth=12,learning_rate=0.16,min_child_samples=50,random_state=2020,n_jobs=8)
    model.fit(X, y)
    importantFeatures = model.feature_importances_
    Values = np.sort(importantFeatures)[::-1] #SORTED
    CriticalValue=np.mean(Values)
    K = importantFeatures.argsort()[::-1][:len(Values[Values>CriticalValue])]
    LGB_ALL_K=pd.concat([y,X.iloc[:,K]],axis=1)
    LGB_ALL_K.to_csv(f+"LGBM_SF_"+str(len(K))+".csv",index=False)
    print("_LGBM_SF_K=",LGB_ALL_K.shape[1])
    print("LGBoosting features selections completed!!!!")
    print("All finished! Good Luck!!!")
    return  LGB_ALL_K.columns
    
def RF_SF(X,y,fileName):#RadomForest Feature Selection Method

    f=fileName+"_"
    print("RandomForestClassifier Selection.....")
    model = RandomForestClassifier(n_estimators=888,min_samples_leaf,random_state=2020,n_jobs=8)
    model.fit(X, y)
    importantFeatures = model.feature_importances_
    Values = np.sort(importantFeatures)[::-1] #SORTED
    CriticalValue=np.mean(Values)
    K = importantFeatures.argsort()[::-1][:len(Values[Values>CriticalValue])]
    RF_ALL_K=pd.concat([y,X.iloc[:,K]],axis=1)
    RF_ALL_K.to_csv(f+"RF_SF_"+str(len(K))+".csv",index=False)
    print("_RF_SF_K=",LGB_ALL_K.shape[1])
    print("LGBoosting features selections completed!!!!")
    print("All finished! Good Luck!!!")
    return  RF_ALL_K.columns

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser('Script for embedding fasta format sequences using a saved embedding model. Saves embeddings as CSV file.')
    parser.add_argument('-SF', help='Feature Selection Method: RF or LGBM')
    parser.add_argument('-i', help='input CSV file')
    parser.add_argument('-o', help='output CSV file')
    args = parser.parse_args()
    
    data_in=pd.read_csv(args.i,header=0)
    data,Label = get_data(data_in)
    if args.o=="SF":
        KCOL=RF_SF(data,Label,args.o)
       
    if args.o=="LGBM":
        KCOL=LGBM_SF(data,Label,args.o)
        
    print("GOOD LUCK！")
    
    
    