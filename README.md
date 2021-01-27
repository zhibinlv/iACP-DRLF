# iACP-DRLF<br>
Anti-Cancer Peptide Prediction with Deep Representation Learning Features
#How cite the paper?<b>
<div id="refer-anchor-1"></div>
- [1] [Zhibin Lv †, Feifei Cui †, Quan Zou , Lichao Zhang and Lei Xu,Anticancer peptides prediction with deep
representation learning features,Briefings in Bioinformatics, 00(00), 2021, doi: 10.1093/bib/bbab008](http://xueshu.baidu.com/)


This repository contains the source code and links to the data and pretrained embedding models accompanying the iACP-DRLF paper: Anti-Cancer Peptide Prediction with Deep Representation Learning Features

# A GPU like NVIDIA RTX2060 is required.

# Setup and dependencies

Install in Ubuntu Linux 18.04

## Download from http://public.aibiochem.net/iACP-DRLF/

1.  *cd iACP-DRLF

2. *pip install -r pip install -r requirements.txt

3. OK. It could run the python script now.

## Install from git hub 


1. *git clone https://github.com/zhibinlv/iACP-DRLF.git *

2. *cd iACP-DRLF

3. *pip install -r pip install -r requirements.txt

4. *wget bergerlab-downloads.csail.mit.edu/bepler-protein-sequence-embeddings-from-structure-iclr2019/pretrained_models.tar.gz

    *tar -xzvf pretrained_models.tar.gz

    *mv ./pretrained_models/ssa_L1_100d_lstm3x512_lm_i512_mb64_tau0.5_lambda0.1_p0.05_epoch100.sav ./src/PretrainedModel/SSA_embed.model
    
    or you could downloand SSA_embed.model from http://public.aibiochem.net/iACP-DRLF/src/PretrainedModel/SSA_embed.model

7. OK. It could run the python script now.

# Brief tutorial

1. To validate the paper independent test, run the following code.

    
   <font color=red> *python test.py*</font>
   
  ![image](https://github.com/zhibinlv/iACP-DRLF/blob/main/img/Test01.PNG)
  
 2. To use iACP-DRLF

   <font color=red>*python -m {A or M} -i {sequences in FASTA format} -o {output a CSV file}* </font>
   
 >> A is for Alternate dataset trained model
 
 >> M is for Main dataset trained model 
 
 
 
