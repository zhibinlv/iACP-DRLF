from __future__ import print_function,division

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
from src.alphabets import Uniprot21
import src.fasta as fasta
import src.models.sequence
#from jax_unirep import get_reps
from tape import UniRepModel,TAPETokenizer

def unstack_lstm(lstm):
    device = next(iter(lstm.parameters())).device

    in_size = lstm.input_size
    hidden_dim = lstm.hidden_size
    layers = []
    for i in range(lstm.num_layers):
        layer = nn.LSTM(in_size, hidden_dim, batch_first=True, bidirectional=True)
        layer.to(device)

        attributes = ['weight_ih_l', 'weight_hh_l', 'bias_ih_l', 'bias_hh_l']
        for attr in attributes:
            dest = attr + '0'
            src = attr + str(i)
            getattr(layer, dest).data[:] = getattr(lstm, src)
             
            dest = attr + '0_reverse'
            src = attr + str(i) + '_reverse'
            getattr(layer, dest).data[:] = getattr(lstm, src)
             
        layer.flatten_parameters()
        layers.append(layer)
        in_size = 2*hidden_dim
    return layers

def embed_stack(x, lm_embed, lstm_stack, proj, include_lm=True, final_only=False):
    zs = []
    
    x_onehot = x.new(x.size(0),x.size(1), 21).float().zero_()
    x_onehot.scatter_(2,x.unsqueeze(2),1)
    zs.append(x_onehot)
    
    h = lm_embed(x)
    if include_lm and not final_only:
        zs.append(h)

    if lstm_stack is not None:
        for lstm in lstm_stack:
            h,_ = lstm(h)
            if not final_only:
                zs.append(h)
        h = proj(h.squeeze(0)).unsqueeze(0)
        zs.append(h)

    z = torch.cat(zs, 2)
    return z


def embed_sequence(x, lm_embed, lstm_stack, proj, include_lm=True, final_only=False
                  ,  pool='none', use_cuda=False):

    if len(x) == 0:
        return None

    alphabet = Uniprot21()
    x = x.upper()
    # convert to alphabet index
    x = alphabet.encode(x)
    x = torch.from_numpy(x)
    if use_cuda:
        #x = x.cuda()
        x = x.to(DEVICE) 

    # embed the sequence
    with torch.no_grad():
        x = x.long().unsqueeze(0)
        z = embed_stack(x, lm_embed, lstm_stack, proj
                       , include_lm=include_lm, final_only=final_only)
        # pool if needed
        z = z.squeeze(0)
        if pool == 'sum':
            z = z.sum(0)
        elif pool == 'max':
            z,_ = z.max(0)
        elif pool == 'avg':
            z = z.mean(0)
        z = z.cpu().numpy()

    return z


def load_model(path, use_cuda=False):
    encoder = torch.load(path)
    encoder.eval()

    if use_cuda:
        #encoder.cuda()
        encoder=encoder.to(DEVICE) 

    if type(encoder) is src.models.sequence.BiLM:
        # model is only the LM
        return encoder.encode, None, None

    encoder = encoder.embedding

    lm_embed = encoder.embed
    lstm_stack = unstack_lstm(encoder.rnn)
    proj = encoder.proj

    return lm_embed, lstm_stack, proj


def DRLF_Embed(fastaFile,outFile,device=-2):
    
    path = fastaFile
    count = 0
    SSAEMB_=[]
    UNIREPEB_=[]
    ##read Fasta File
    inData=fasta.fasta2csv(path)
    Seqs=inData["Seq"]
    
    PID_=[]
    ##SSA Embedding
     
    print("SSA Embedding...") 
    lm_embed, lstm_stack, proj=load_model("./src/PretrainedModel/SSA_embed.model", use_cuda=True)
   
    with open(path, 'rb') as f:
        for name,sequence in fasta.parse_stream(f):
            
            pid =str( name.decode('utf-8'))
            if len(sequence) == 0:
                print('# WARNING: sequence', pid, 'has length=0. Skipping.', file=sys.stderr)
                continue
            
            PID_.append(pid)
            
            z = embed_sequence(sequence, lm_embed, lstm_stack, proj
                                  ,  final_only=True
                                  , pool='avg', use_cuda=True)
            
            SSAEMB_.append(z)
            count += 1
            print(sequence,'# {} sequences processed...'.format(count), file=sys.stderr, end='\r')
    print("SSA embedding finished@")
    
     
    ssa_feature=pd.DataFrame(SSAEMB_)
    col=["SSA_F"+str(i+1) for i in range(0,121)]
    ssa_feature.columns=col
    
    
    print("UniRep Embedding...")
    print("Loading UniRep Model...",    file=sys.stderr, end='\r')
    
    model = UniRepModel.from_pretrained('babbler-1900')
    model=model.to(DEVICE)
    tokenizer = TAPETokenizer(vocab='unirep') 
    
     
    count =0
    PID_=inData["PID"]
    
    for sequence in Seqs:
            
        if len(sequence) == 0:
            print('# WARNING: sequence', pid, 'has length=0. Skipping.', file=sys.stderr)
            continue
          
        
        with torch.no_grad():
            token_ids = torch.tensor([tokenizer.encode(sequence)])
            token_ids = token_ids.to(DEVICE)
            output = model(token_ids)
            unirep_output = output[0]
            #print(unirep_output.shape)
            unirep_output=torch.squeeze(unirep_output)
            #print(unirep_output.shape)
            unirep_output= unirep_output.mean(0)
            unirep_output = unirep_output.cpu().numpy()
             
           # print(sequence,len(sequence),unirep_output.shape)
            UNIREPEB_.append(unirep_output.tolist())      
            count += 1
            print(sequence,'# {} sequences processed...'.format(count), file=sys.stderr, end='\r')
          
    unirep_feature=pd.DataFrame(UNIREPEB_)
     
    
    col=["UniRep_avg_F"+str(i+1) for i in range(0,1900)]
    unirep_feature.columns=col
    print("UniRep Embedding Finished@!")
    Features=pd.concat([ssa_feature,unirep_feature],axis=1)
    Features.index=PID_
    Features.to_csv(outFile)
    print("Getting Deep Representation Learning Features is done.")
    
    return Features,inData



if __name__ == '__main__':
    import argparse
    import time
    parser = argparse.ArgumentParser('Script for embedding fasta format sequences using a saved embedding model. Saves embeddings as CSV file.')

    parser.add_argument('-i', help='sequences to embed in fasta format')
    parser.add_argument('-o', help='path to saved embedding CSV file')
    args = parser.parse_args()
    T0=time.time()
    DRLF_Embed(args.i,args.o)
    print("It takes ",(time.time()-T0)/60,"mins")
    
    
     
    




