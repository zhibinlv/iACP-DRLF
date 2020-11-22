import pandas as pd
import src.getFeatures as getFeatures
import joblib 
import argparse
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def predict(Model_num,inFasta,outFile):
    Model_num=str(Model_num)
    inFasta=str(inFasta)
    outFile=str(outFile)
    Features,seqs=getFeatures.DRLF_Embed(inFasta,outFile.replace(".csv","")+"Feature.csv")
    print(seqs.shape)
    print(Features.shape)
    
    if Model_num=="M":
        model=joblib.load("./src/PretrainedModel/MAINBestLGBM148.joblib")
        print("MainDataset Trained Model:",model)

        scale=joblib.load("./src/PretrainedModel/MainStandardScaler633.joblib")
        trainFeature=pd.read_csv("./src/PretrainedModel/MainSetFeature.csv",header=None)
        X=Features[trainFeature.iloc[0,:]]
        X=scale.transform(X)
        X=X[:,:148]
        y_pred=model.predict(X)
        y_pred_prob=model.predict_proba(X)
        
        
    if Model_num=="A":
        model=joblib.load("./src/PretrainedModel/ALTBestLGBM129.joblib")
        print("AlternateDataset Trained Model:",model)
        scale=joblib.load("./src/PretrainedModel/AltSetStandardScaler493.joblib")
        trainFeature=pd.read_csv("./src/PretrainedModel/AltFeature.csv",header=None)
        X=Features[trainFeature.iloc[0,:]]
        X=scale.transform(X)
        X=X[:,:129]
        y_pred=model.predict(X)
        y_pred_prob=model.predict_proba(X)

     
    
    

    df_out=pd.DataFrame(np.zeros((y_pred_prob.shape[0],5)),columns=["Index","Seq","Prediction","Pred_Label","isACP_Probability"])

    for i in range(y_pred.shape[0]):
        df_out.iloc[i,0]=i+1
        df_out.iloc[i,1]=seqs["Seq"][i]
        if y_pred[i]==1:
          df_out.iloc[i,2]="ACP"
          df_out.iloc[i,3]=1
        if y_pred[i]==0:
          df_out.iloc[i,2]="non-ACP"
          df_out.iloc[i,3]=0
        df_out.iloc[i,4]=y_pred_prob[i,1]
        df_out.to_csv(outFile.replace(".csv","")+"PredResults.csv",index=False)
    print(df_out.shape)
    print(df_out.head())
    return df_out
    
if __name__ == '__main__':
   
    parser = argparse.ArgumentParser('Script for AntiCancer Peptide identification by Deep Representation Learning Features.')
    parser.add_argument("-m",help="M for model trained on mainDataset, A for model trained on alternateDataset")
    parser.add_argument('-i', help='sequences in fasta format')
    parser.add_argument('-o', help='path to save prediction results in a CSV file')
    args = parser.parse_args()
    T0=time.time()
    predict(args.m,args.i,args.o)
    #df_out=predict(args.m,args.i,args.o)
    print("It takes",(time.time()-T0)/60,"mins!")



