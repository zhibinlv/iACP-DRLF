# iACP-DRLF
Anti-Cancer Peptide Prediction with Deep Representation Learning Features

This repository contains the source code and links to the data and pretrained embedding models accompanying the iACP-DRLF paper: Anti-Cancer Peptide Prediction with Deep Representation Learning Features

# A GPU like NVIDIA RTX2060 is required.

# Setup and dependencies

Install in Ubuntu Linux 18.04


1. git clone https://github.com/zhibinlv/iACP-DRLF.git 

2. cd iACP-DRLF

3. pip install -r pip install -r requirements.txt

4. wget bergerlab-downloads.csail.mit.edu/bepler-protein-sequence-embeddings-from-structure-iclr2019/pretrained_models.tar.gz

5. tar -xzvf pretrained_models.tar.gz

6. mv ./pretrained_models/ssa_L1_100d_lstm3x512_lm_i512_mb64_tau0.5_lambda0.1_p0.05_epoch100.sav ./src/PretrainedModel/SSA_embed.model

7. OK. It could run the python script now.

# Brief tutorial

1. To validate the paper independent test, run the following code.

   python test.py
   
  ![image](./img/Test.png)
 
