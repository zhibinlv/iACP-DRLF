import pandas as pd
import numpy as np
from iACP_DRLF import predict
from sklearn.metrics import roc_curve,accuracy_score,auc,matthews_corrcoef,confusion_matrix
import warnings
warnings.filterwarnings("ignore")

def getMetrics(y_true,y_pred,y_proba):
    ACC= accuracy_score(y_true,y_pred)
    MCC= matthews_corrcoef(y_true,y_pred)
    CM=confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = CM.ravel()
    Sn=tp/(tp+fn)
    Sp=tn/(tn+fp)
    FPR,TPR,thresholds_ =roc_curve(y_true, y_proba)
    AUC=auc(FPR, TPR)
    
    Results=np.array([ACC,MCC,Sn,Sp,AUC]).reshape(-1,5)
    print(Results.shape)
    Metrics_=pd.DataFrame(Results,columns=["ACC","MCC","Sn","Sp","AUC"])
   
    return Metrics_
    
  
AltTestLabel=pd.read_csv("./test/AltTestLabel.csv",header=0)
MainTestLabel=pd.read_csv("./test/MainTestLabel.csv",header=0)
AltTest="./test/ACP20AltTest.fasta"
MainTest="./test/ACP20mainTest.fasta"

 
pred_alt=predict("A",AltTest,"ACP20_AltTest")
m=getMetrics(AltTestLabel,pred_alt.iloc[:,3],pred_alt.iloc[:,4])
print("Independent Testing Results for Alternate Dataset Trained Model\n")
print(m)
print("\n")
m.to_csv("./test/MainTest_MetricResults.csv")
    
    
pred_main=predict("M",MainTest,"ACP20_MainTest")
m=getMetrics(MainTestLabel,pred_main.iloc[:,3],pred_main.iloc[:,4])
print("Independent Testing Results for MainDataset Trained Model\n")
print(m)
print("\n")
m.to_csv("./test/AltTest_MetricResults.csv")
    

    