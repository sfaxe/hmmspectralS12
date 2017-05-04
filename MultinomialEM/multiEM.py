import os
import time
import warnings
import numpy as np
import scipy.sparse.linalg
import scipy.sparse
from sklearn.metrics.cluster import v_measure_score
from hmmlearn import hmm
def dictEncode(dict, term):
    if dict.get(term) is not None:
       return dict.get(term)
    else:
       dict[term]=len(dict)
       return dict.get(term)
def xmlProcess(xmlFile,tag_dict,token_dict,a=3,b=1,c=2):
    tree = ET.parse(xmlFile)
    root = tree.getroot()
    
    senList=[]
    encodeList=[]
    for neighbor in root.iter():
       dict=neighbor.attrib
       if dict.get('pos'):
          pos=dict.get('pos')
          lemma=dict.get('lemma')
          beginID=dict.get("begin")
          senList.append((pos, lemma,int(beginID)))
          encodeList.append((dictEncode(tag_dict, pos),dictEncode(token_dict, lemma),int(beginID)))
    senList=sorted(senList, key=lambda x: x[2])
    encodeList=sorted(encodeList, key=lambda x: x[2])
    #print senList
    #print encodeList 
    tok=[x[1] for x in encodeList]
    sub_ab=[]
    sub_ac=[]
    sub_abc=[]
    #switch index for 0-based
    a=a-1
    b=b-1
    c=c-1
    for i in range(len(tok)-max(a,b)):
        sub_ab.append([tok[i+a],tok[i+b]])
    for i in range(len(tok)-max(a,c)):
        sub_ac.append([tok[i+a],tok[i+c]])    
    for i in range(len(tok)-max(a,b,c)):
        sub_abc.append([tok[i+a],tok[i+b],tok[i+c]]) 
    return(tok,[x[0] for x in encodeList],sub_ab,sub_ac,sub_abc)

def txtProcess(txtLine,tag_dict,token_dict,a=3,b=1,c=2):
    
    senList=[]
    encodeList=[]
    txtList=txtLine.split()
    for txt in txtList:
       tok_tag=txt.split("_")
       k=len(tok_tag)
       if k==2:
          pos=tok_tag[1]
          lemma=tok_tag[0]
       else:
          pos=tok_tag[k-1]
          lemma="_".join(tok_tag[0:k-1])
       senList.append(lemma)
       encodeList.append((dictEncode(tag_dict, pos),dictEncode(token_dict, lemma)))
    
    tok=[x[1] for x in encodeList]
    #print tok
    sub_ab=[]
    sub_ac=[]
    sub_abc=[]
    #switch index for 0-based
    a=a-1
    b=b-1
    c=c-1
    for i in range(len(tok)-max(a,b)):
        sub_ab.append([tok[i+a],tok[i+b]])
    for i in range(len(tok)-max(a,c)):
        sub_ac.append([tok[i+a],tok[i+c]])    
    for i in range(len(tok)-max(a,b,c)):
        sub_abc.append([tok[i+a],tok[i+b],tok[i+c]]) 
    return([str(x[1]) for x in encodeList],tok,[x[0] for x in encodeList],sub_ab,sub_ac,sub_abc)



def P_matrix_1d(listCount,d):
    w=np.zeros(d)    
    N=float(len(listCount))
    for x in listCount:
        w[int(x)]=w[int(x)]+1
    return np.true_divide(w, N)

def P_matrix_2d(listCount,d):
    w=scipy.sparse.dok_matrix((d,d), dtype=np.dtype('Float64')) 
    Ninv=float(1)/float(len(listCount))
    for x in listCount:
        w[int(x[0]),int(x[1])]=w[int(x[0]),int(x[1])]+Ninv
    return w

def P_matrix_3d(listCount,d):
    #w=np.zeros((d,d,d))   
    S=[]
    for i in range (d):
        S.append(scipy.sparse.dok_matrix((d,d), dtype=np.dtype('Float64')))
    #1/N
    Ninv=float(1)/float(len(listCount))
    for x in listCount:
        w=S[int(x[0])]
        w[int(x[1]),int(x[2])]=w[int(x[1]),int(x[2])]+Ninv
    return S

def randMatrixGen(k):
    A=np.random.rand(k,k)
    theta, R = np.linalg.qr(A)
    Rprime=np.diag(np.sign(np.diag(R)))
    theta=np.dot(theta, Rprime)
    thetaDet=np.linalg.det(theta)
    if thetaDet<0:
       temp = np.copy(theta[:, 0])
       theta[:, 0]=theta[:, 1]
       theta[:, 1]=temp
    return theta

# return U1t*P*U2
def sandwichP(U1,P,U2):
    return np.dot(np.transpose(U1),P.dot(U2))
#tensor P123(U3theta_i),where transpose(theta_i) is the i th row vector of random matrix theta
#tensor W=T(ita) (ita is 1D) is defined as W_ij=sum_over_x(ita_x T_ijx)
#input is U3t
#dimension check: U3 is (d,k),transpose(theta_i) is (1,k) so U3theta_i is (d,1)
#so w should be (d,d) (P123 is (d,d,d)
def P_entry(T,theta,i,U3,d):
    i_row=np.copy(theta[i,:])
    ita=np.dot(U3,np.transpose(i_row))
    P=scipy.sparse.dok_matrix((d,d), dtype=np.dtype('Float64')) 
    for i in range(d):
        slice=T[i]
        P[i,:]=slice.dot(ita)
    return P

# return diag(B_123(U3theta_i))
# to save computation cost,precompute W=inverse(U1T*P12*U2)    
def eigB(P123,theta,U3,W,d,U1,U2,i=0):
    B=np.dot(sandwichP(U1,P_entry(P123,theta,i,U3,d),U2),W)
    return np.linalg.eig(B)

def diagB(P123,theta,i,U3,W,d,R,U1,U2):
    B=np.dot(sandwichP(U1,P_entry(P123,theta,i,U3,d),U2),W)
    return np.diag(np.dot(np.dot(np.linalg.inv(R),B),R))

def constructL(k,P123,theta,U3,W,d,U1,U2):
    L=np.zeros((k,k))
    v,R=eigB(P123,theta,U3,W,d,U1,U2)
    for i in range(k):
       L[i,:]=diagB(P123,theta,i,U3,W,d,R,U1,U2)
    return (L,R)

def negativeHandlingAbs(T):
    nrow,ncol=T.shape
    print ncol
    Q=np.absolute(np.real(T))
    for j in range(ncol):
        sumCol=np.sum(Q[:,j])
        print sumCol
        Q[:,j]=np.true_divide(Q[:,j],sumCol)
        print Q[:,j]
    #print Q
    return Q

def negativeHandlingArrayAbs(T):
    Q=np.absolute(np.real(T))
    sumCol=np.sum(Q)
    Q=np.true_divide(Q, sumCol)
    return Q


def negativeHandling(T):
    nrow,ncol=T.shape
    print ncol
    K=np.real(T)
    output=np.copy(K)
    small=1e-40*np.ones((nrow,ncol))
    pickNeg=np.absolute(K)-K+small
    pickPos=np.absolute(K)+K+small
    for j in range(ncol):
        if np.sum(K[:,j])>=0:
           sumCol=np.sum(pickPos[:,j])
           output[:,j]=np.true_divide(pickPos[:,j],sumCol)
        else:
           sumCol=np.sum(pickNeg[:,j])
           output[:,j]=np.true_divide(pickNeg[:,j],sumCol)
    return output

def negativeHandlingArray(T):
    K=np.real(T)
    output=np.copy(K)
    small=1e-40*np.ones(T.shape)
    pickNeg=np.absolute(K)-K+small
    pickPos=np.absolute(K)+K+small
    
    if np.sum(K)>=0:
           sumCol=np.sum(pickPos)
           output=np.true_divide(pickPos,sumCol)
    else:
           sumCol=np.sum(pickNeg)
           output=np.true_divide(pickNeg,sumCol)
    return output

def hmmSpectral(pi1,T1,O1,Xseq,Zseq,n_feat,n_comp,lenInfo,extra_iter):
    model=hmm.MultinomialHMM(n_components=n_comp, startprob_prior=pi1, transmat_prior=T1, algorithm='viterbi', random_state=None, n_iter=extra_iter, tol=0.000001, verbose=False, params='ste', init_params='')
    model.n_features=n_feat
    
    model.startprob_ =pi1
    model.transmat_ = T1
    model.emissionprob_=O1   
    if extra_iter>0:
       model.fit(np.atleast_2d(Xseq).T,lenInfo)
    hidden_states = model.predict(np.atleast_2d(Xseq).T,lenInfo)
    q=v_measure_score(Zseq,hidden_states)
    print "spectral: iter=",extra_iter," v info=",q

def hmmNoSpectral(Xseq,Zseq,n_feat,n_comp,lenInfo,iter):
    model=hmm.MultinomialHMM(n_components=n_comp, algorithm='viterbi', random_state=None, n_iter=iter, tol=0.000001, verbose=False, params='ste', init_params='ste')
    model.n_features=n_feat
    model.fit(np.atleast_2d(Xseq).T,lenInfo)
    hidden_states = model.predict(np.atleast_2d(Xseq).T,lenInfo)
    q=v_measure_score(Zseq,hidden_states)
    return q

def PreTXT():
    tag_dict = {}
    token_dict={}
    X=[]
    Z=[]
    list_ab=[]
    list_ac=[]
    list_abc=[]
    list_P1=[]
    sentences=[]
    for file in os.listdir("./masc/"):
        if file.endswith(".txt"):
           f="./masc/"+file
           with open(f, "r") as ins:
            for line in ins:
               if len(line)>2:   
                  if line.count("_")>=3:
                     #sequence of observable X variables is generated by a sequence of internal hidden states Z.
                     sent,Xsample, Zsample,sub_ab,sub_ac,sub_abc=txtProcess(line,tag_dict,token_dict)
                     sentences.append(sent)
                     print line
                     print sent
                     print Xsample
                     X.append(Xsample)
                     Z.append(Zsample)
                     list_P1.append(Xsample[0])
                     list_ab.extend(sub_ab)
                     list_ac.extend(sub_ac)
                     list_abc.extend(sub_abc)
    return (sentences,X,Z,len(token_dict),len(tag_dict),list_P1,list_ab,list_ac,list_abc)



(sentences,X,Z,len_token,len_tag,list_P1,list_ab,list_ac,list_abc)=PreTXT()


print "Dictionary size(unique tokens): ",len_token
print 'number of hidden states (POS tags) are',len_tag


try:
    with open("MultiEMBenchmark.txt", "r") as file:
        pass
except IOError as e:
    with open("MultiEMBenchmark.txt", "w") as file:
        file.write("maxIter,FullTime,V_measure,"+"\n")

lenInfo=[]
for t in range(len(Z)):
    lenInfo.append(len(Z[t]))
# Make an HMM instance and execute fit
Xseq=np.concatenate(X, axis=0)
Zseq=np.concatenate(Z, axis=0)
#Zseq=Zseq.astype(np.float64)
#Xseq=Xseq.astype(np.float64)
maxIter=[1,2,5,10]
for maxI in maxIter:
 for i in range(5):
   start_time = time.time()
   with warnings.catch_warnings():
       warnings.simplefilter("ignore")
       q=hmmNoSpectral(Xseq,Zseq,len_token,len_tag,lenInfo,maxI)

   embedTimeS=time.time() - start_time
   with open("MultiEMBenchmark.txt", "a") as f: 
       f.write(str(maxI)+","+str(embedTimeS)+","+str(q)+",\n") 


