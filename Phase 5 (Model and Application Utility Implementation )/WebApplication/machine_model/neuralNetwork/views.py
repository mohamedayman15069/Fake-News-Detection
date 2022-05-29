from django.shortcuts import render

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
from gensim.models import Word2Vec
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
from nltk.stem.porter import PorterStemmer


# The Dictinary for contractions
contractions = { 
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"i'd": "i had / i would",
"i'd've": "i would have",
"i'll": "i shall / i will",
"i'll've": "i shall have / i will have",
"i'm": "i am",
"im": "i am",
"i've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}


class NewsTitle:
   
    def __init__(self,title):
        self.title = title
        words = set(nltk.corpus.words.words())
        nltk.download('words')

        
    def cont_to_exp(self, x):
        if type(x) is str:
            for key in contractions:
                value = contractions[key]
                x=x.replace(key, value)
            return x
        else:
            return x
        
    def tokenization(self, x):
        return x.split()
    
    def stemming(self, x):
        stems = []
        porter = PorterStemmer()
        for t in x:
            lemma=str(porter.stem(t))
            if lemma == '-PRON-' or lemma == 'be':
                lemma=t
            stems.append(lemma)
        return stems
    def make_to_base(self, x):
        x_list=[]
        nlp= spacy.load('en_core_web_sm')
        doc=nlp(" ".join(x))
        for token in doc:
            lemma= str(token.lemma_)
            if lemma == '-PRON-' or lemma == 'be':
                lemma=token.text
            x_list.append(lemma)
        return x_list

    def preprocessing(self, W2V_model):
        self.title= self.cont_to_exp(self.title)
        self.title= self.title.lower()
        self.title= " ".join([t for t in self.title.split() if t not in STOP_WORDS])
        print(self.title)
        self.title= self.tokenization( self.title)
        self.title= self.stemming(self.title)
        self.title= self.make_to_base(self.title)
        #Check if exists 
        self.title= np.mean([W2V_model.wv[token] for token in self.title if token in W2V_model.wv],axis=0).tolist()
        #print(self.title)
        return self.title

'''
nltk.download('words')
words = set(nltk.corpus.words.words())
porter = PorterStemmer()
nlp= spacy.load('en_core_web_sm')
'''

'''
    Forwarding the Model with the user input
'''
# Create your views here.
def Sigmoid(Z):
    m=np.max(Z,axis=0).reshape(-1,1)
    Z=Z-m.T
#     print(Z.shape)
    
    A=np.exp(Z)/np.sum(np.exp(Z),axis=0,keepdims=True)
#     print(f'A{ A.shape}')
    return A 

def Relu(Z):
    return np.maximum(Z,0)

def Forward_Prob(Xt,Wts,NN):
    # we need to store W,b,inputs,outputs,activations for each layer for each node
    parameters=[]
    Ai=Xt #current layer output
    #print("SHAPE", Xt.shape)
    Ap=Ai #previous layer output (current layer input)
    for i in range(1,(NN)-1):
        W,b=Wts['W'+str(i)],Wts['b'+str(i)]
        Ap=Ai
        Z=np.dot(W,Ap.T)+b
       # print("W",W.shape)
       # print("AP", Ap.T.shape)
       # print("Z: ", Z.shape)
        actv=Relu(Z)
        Ai=actv.T
       # print(Ai.shape,Ap.shape,actv.shape)
        #print(Wts['b'+str(i)])
        parameters.append((W,b,Ap,actv,Z))
    W,b=Wts['W'+str((NN)-1)],Wts['b'+str((NN)-1)]
    #print("W",Ai.shape)

    Ap=Ai
    Z=np.dot(W,Ap.T)+b
    actv=Sigmoid(Z) #np.exp(Z) / np.sum(np.exp(Z), axis=0)#
    Ai=actv.T
   # print(Ai.shape,Ap.shape,actv.shape)
    #print(Wts['b'+str(i)])
    parameters.append((W,b,Ap,actv,Z))
    Ao=Ai #output of last layer=last Ai   
    #print(len(parameters))
    return parameters,Ao 

def listToString(s): 
    
    # initialize an empty string
    str1 = "" 
    
    # traverse in the string  
    for ele in s: 
        str1 += ele  
    
    # return string  
    return str1 
        
#BackProp for the bonus 

# Probgate backwards
def Back_Prob(y_true, Ao, parameters,Wts,NN):
    #dAo = - (np.divide(Ao, y_true))  #last layer derivative
    dZ =(Ao-y_true)
    derivatives={}
#     print(f'dz {dZ}')
    m=y_true.shape[0]
    W,b,X,A,Z=parameters[len(NN)-2]
    
    dW=1.0/m*np.dot(dZ.T,X)
#     print(f' 1st dW {dW}')
    #print(A==Sigmoid(np.dot(W,X.T)+b))
    db = (1/(m)) * np.sum(dZ, axis =0, keepdims = True).T
    derivatives['dW'+str(len(NN)-1)]=dW
    derivatives['db'+str(len(NN)-1)]=db
    #dAp = np.matmul( dZ,W)
   # print(W)
    for i in reversed(range(len(NN)-2)):
        Wp=W
        Xp=X
        #dA=dAp
        W,b,X,A,Z=parameters[i]
        dZp=dZ
        grad=np.ones((A.shape))
        grad[A<0]=0
        dZ=np.dot(dZp,Wp)*grad.T #* (1 - np.power(A.T, 2))# #vectorized implementation of oh in the slides
#         print(f'2d dZ {dZ}')
        m=X.shape[0]
    #    print("HERE",m)
        dW=1.0/m*np.dot(dZ.T,X)
#         print(f'2d dW {dW}')
        db = (1/(m)) * np.sum(dZ, axis =0, keepdims = True).T
#         print(f'2d db {db}')
        #print(dZp.shape,X.shape,Wp.shape,A.shape, db.shape,dZ.shape)
        derivatives['dW'+str(i+1)]=dW
        derivatives['db'+str(i+1)]=db
    return derivatives

#Update the weights 
# 
def update(derivatives,Wts,parameters,learning_rate,NN):
    for i in range(1,len(NN)):
        #print(i)
        Wts['b'+str(i)]=Wts['b'+str(i)]-learning_rate*derivatives['db'+str(i)]
        Wts['W'+str(i)]=Wts['W'+str(i)]-learning_rate*derivatives['dW'+str(i)]
    #print(Wts['W2'])
    return Wts

       
        
    
    
embedding = any
G_weight = any

def enter(request):
    number = 10
    
    file= open(r'''/home/ayman/Django/ML_Project2/machine_model/neuralNetwork/word2vec.model''', 'rb')
    W2V_model=pickle.load(file)
    query = 's'
    if request.method == "GET":
        query = request.GET.get('query')
        
        print(query)
        title= NewsTitle(listToString(query))

        emb = title.preprocessing(W2V_model)
        embedding = emb
        #emb = title.preprocessing(W2V_model)

        #Model

        # Loading the weights 
        w_file= open(r'''/home/ayman/Django/ML_Project2/machine_model/neuralNetwork/Weights.model''', 'rb')
        weights = pickle.load(w_file)
        #print("HEELO")
        G_weight = weights
        params,predicted = Forward_Prob(np.array(emb)[np.newaxis],weights,5)
        idx = (np.argmax(predicted))
        st = ''
        if idx == 0:
            st = 'Fake'
        else: 
            st = 'Real' 

        


        context = {"st": st}
    return render(request, 'enter_data.html', context)


def validate(request):
    query = 's'
     
    if request.method == "GET":
        query = request.GET.get('query')
 
    
    isfake = [0,0]

    if(query != 1):
        isfake[0] = 1
    else: 
        isfake[1] = 1
    
    params, Ao = Forward_Prob(embedding,G_weight,5)
    dv = Back_Prob(isfake, Ao, params,G_weight,5)
    wts = update(dv,G_weight,params,0.1,5)
    G_weight = wts 
    w_file= open(r'''/home/ayman/Django/ML_Project2/machine_model/neuralNetwork/Weights.model''', 'wb')
    pickle.dump(G_weight,w_file)
    w_file.close()
         
     
    context = {"done": "done"}
    return render(request, 'enter_data.html', context)