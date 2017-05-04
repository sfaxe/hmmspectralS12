import xml.etree.ElementTree as ET
import gensim
import os
import numpy as np
import scipy.sparse.linalg
import scipy.sparse
from sklearn.metrics.cluster import v_measure_score
from hmmlearn import hmm
import progressbar as pgb
import warnings
import time

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
    return([str(x[1]) for x in encodeList],tok,[x[0] for x in encodeList],sub_ab,sub_ac,sub_abc)


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


def P_matrix_1d_vec(listCount,d,word_vectors):
    w=np.zeros(d)    
    N=float(len(listCount))
    for x in listCount:
        w=w+word_vectors[int(x)]
    return np.true_divide(w, N)

def searchOuter(outDict,word_vectors,a,b):
    term=(a,b)
    revTerm=(b,a)
    if outDict.get(term) is not None:
       return outDict.get(term)
    else:
       if outDict.get(revTerm) is not None:
          return np.transpose(outDict.get(revTerm))
       else:
          outDict[term]=np.outer(word_vectors[int(a)],word_vectors[int(b)])
       return outDict.get(term)
    
def P_matrix_2d(listCount,d):
    w=scipy.sparse.dok_matrix((d,d), dtype=np.dtype('Float64')) 
    Ninv=float(1)/float(len(listCount))
    for x in listCount:
        w[int(x[0]),int(x[1])]=w[int(x[0]),int(x[1])]+Ninv
    return w

def P_matrix_2d_vec(listCount,d,word_vectors):
    w=np.zeros((d,d))    
    N=float(len(listCount))
    avecList=[word_vectors[int(x[0])] for x in listCount]
    bvecList=[word_vectors[int(x[1])] for x in listCount]
    aDimList=[]
    bDimList=[]
    for i in range(d):
        dA=[vec[i] for vec in avecList]
        dB=[vec[i] for vec in bvecList]
        aDimList.append(np.array(dA))
        bDimList.append(np.array(dB))
    for i in range(d):
        for j in range(d):
            w[i,j]=np.dot(aDimList[i],bDimList[j])
    
    return np.true_divide(w, N)

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

def P_matrix_3d_vec(listCount,d,word_vectors):
       
    S=[]
    for i in range (d):
        S.append(np.zeros((d,d)))
    #1/N
    avecList=[word_vectors[int(x[0])] for x in listCount]
    bvecList=[word_vectors[int(x[1])] for x in listCount]
    cvecList=[word_vectors[int(x[2])] for x in listCount]
    aDimList=[]
    bDimList=[]
    cDimList=[]
    for i in range(d):
        dA=[vec[i] for vec in avecList]
        dB=[vec[i] for vec in bvecList]
        dC=[vec[i] for vec in cvecList]
        aDimList.append(np.array(dA))
        bDimList.append(np.array(dB))
        cDimList.append(np.array(dC))
    Ninv=float(1)/float(len(listCount))
    bar = pgb.ProgressBar()
    for i in bar(range(d)):
        w=S[i]
        for j in range(d):
            for k in range(d):
                w[j,k]=np.dot(np.multiply(aDimList[i],bDimList[j]),cDimList[k])
    
        S[i]=np.true_divide(w, float(len(listCount)))
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

def diagB(P123,theta,i,U3,W,d,R,Rinv,U1,U2):
    dok=scipy.sparse.dok_matrix(sandwichP(U1,P_entry(P123,theta,i,U3,d),U2))

    B=dok.dot(W)
    
    return np.diag(np.dot(Rinv.dot(B),R))

def constructL(k,P123,theta,U3,W,d,U1,U2):
    L=np.zeros((k,k))
    v,R=eigB(P123,theta,U3,W,d,U1,U2)
    Rinv=scipy.sparse.dok_matrix(np.linalg.inv(R))
    bar = pgb.ProgressBar()
    for i in bar(range(k)):
       L[i,:]=diagB(P123,theta,i,U3,W,d,R,Rinv,U1,U2)
    return (L,R)

def negativeHandlingAbs(T):
    nrow,ncol=T.shape
    Q=np.absolute(np.real(T))
    for j in range(ncol):
        sumCol=np.sum(Q[:,j])
        Q[:,j]=np.true_divide(Q[:,j],sumCol)
        
    return Q

def negativeHandlingArrayAbs(T):
    Q=np.absolute(np.real(T))
    sumCol=np.sum(Q)
    Q=np.true_divide(Q, sumCol)
    return Q


def negativeHandling(T):
    nrow,ncol=T.shape
    
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
def negativeHandlingO(T):
    nrow,ncol=T.shape
    
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
    
    return output,np.true_divide((T<0.0).sum(),(nrow*ncol))




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

def hmmGaussianSpectral(pi1,T1,O1,Xseq,Zseq,n_feat,n_comp,lenInfo,extra_iter=0):
    if extra_iter==0:
       model=hmm.GaussianHMM(n_components=n_comp, covariance_type='diag', min_covar=0.000001, startprob_prior=pi1, transmat_prior=T1, means_prior=O1, algorithm='viterbi', random_state=None, n_iter=1, tol=0.000001, verbose=False, params='c', init_params='c') 
    else:
       model=hmm.GaussianHMM(n_components=n_comp, covariance_type='diag', min_covar=0.000001, startprob_prior=pi1, transmat_prior=T1, means_prior=O1, algorithm='viterbi', random_state=None, n_iter=1, tol=0.000001, verbose=False, params='stmc', init_params='c')
    model.n_features=n_feat   
    model.startprob_ =pi1
    model.transmat_ = T1
    model.means_=O1   
    model.fit(Xseq,lenInfo)
    hidden_states = model.predict(Xseq,lenInfo)
    q=v_measure_score(Zseq,hidden_states)
    return q

def hmmGaussianNoSpectral(Xseq,Zseq,n_feat,n_comp,lenInfo,extra_iter):
    model=hmm.GaussianHMM(n_components=n_comp, covariance_type='diag', min_covar=0.000001, algorithm='viterbi', random_state=None, n_iter=extra_iter, tol=0.000001, verbose=False, params='stmc', init_params='stmc') 
    model.n_features=n_feat   
    if extra_iter>0:
       model.fit(Xseq,lenInfo)
    hidden_states = model.predict(Xseq,lenInfo)
    q=v_measure_score(Zseq,hidden_states)
    print "no spectral: iter=",extra_iter," v info=",q
def hmmNoSpectral(Xseq,Zseq,n_feat,n_comp,lenInfo,iter):
    model=hmm.MultinomialHMM(n_components=n_comp, algorithm='viterbi', random_state=None, n_iter=iter, tol=0.000001, verbose=False, params='ste', init_params='ste')
    model.n_features=n_feat
    model.fit(np.atleast_2d(Xseq).T,lenInfo)
    hidden_states = model.predict(np.atleast_2d(Xseq).T,lenInfo)
    q=v_measure_score(Zseq,hidden_states)
    print "em iter=",iter," v info=",q

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
                     #print line
                     #print sent
                     #print Xsample
                     X.append(Xsample)
                     Z.append(Zsample)
                     list_P1.append(Xsample[0])
                     list_ab.extend(sub_ab)
                     list_ac.extend(sub_ac)
                     list_abc.extend(sub_abc)
    return (sentences,X,Z,len(token_dict),len(tag_dict),list_P1,list_ab,list_ac,list_abc)

def PreXML():
    tag_dict = {}
    token_dict={}
    X=[]
    Z=[]
    list_ab=[]
    list_ac=[]
    list_abc=[]
    list_P1=[]
    sentences=[]
    for file in os.listdir("./xmlData/"):
        if file.endswith(".xml"):
           #sequence of observable X variables is generated by a sequence of internal hidden states Z.
               sent,Xsample,Zsample,sub_ab,sub_ac,sub_abc=xmlProcess("./xmlData/"+file,tag_dict,token_dict)
               sentences.append(sent)
               X.append(Xsample)
               Z.append(Zsample)
               list_P1.append(Xsample[0])
               list_ab.extend(sub_ab)
               list_ac.extend(sub_ac)
               list_abc.extend(sub_abc)
    
    return (sentences,X,Z,len(token_dict),len(tag_dict),list_P1,list_ab,list_ac,list_abc)


def embedEncode(sentences, dim,len_token,lenZ,X,sg=False):
    if sg:
       model = gensim.models.Word2Vec(sentences, workers=4, size=dim, min_count=1, window=1, sample=1e-16,sg=1)
    else:
       model = gensim.models.Word2Vec(sentences, workers=4, size=dim, min_count=1, window=1, sample=1e-16)
    word_vectors=[]
    for i in range(len_token):
       w1=str(i)
       word_vectors.append(model[w1])
    del model
    X_new=[]
    for t in range(len(Z)):
       W=X[t]
       uList=[]
       for u in W:
          uList.append(word_vectors[u])
       X_new.append(uList)
    return (X_new,word_vectors)
#(X,Z,len_token,len_tag,list_P1,list_ab,list_ac,list_abc)=PreTXT()
(sentences,X,Z,len_token,len_tag,list_P1,list_ab,list_ac,list_abc)=PreTXT()
#(sentences,X,Z,len_token,len_tag,list_P1,list_ab,list_ac,list_abc)=PreXML()

print "Dictionary size(unique tokens): ",len_token

try:
    with open("GauSpecBenchmark.txt", "r") as file:
        pass
except IOError as e:
    with open("GauSpecBenchmark.txt", "w") as file:
        file.write("ExtraDim,Dim,Skip-gram?,Embedding_BuiltTime,Matrix_SolvingTime,Covar_estimateTime,FullTime,NegRatio,V_measure"+"\n")

lenInfo=[]

for t in range(len(Z)):
    lenInfo.append(len(Z[t]))
# Make an HMM instance and execute fit
Zseq=np.concatenate(Z, axis=0)

extraDimList=[1,5,10,20,40,80]
for dimFactor in range(len(extraDimList)):
   for sgFactor in range(2):
     if sgFactor==0:
       sgSign=False
     else:
       sgSign=True
     for iterati in range(5):
        extra_dim=extraDimList[dimFactor]
        dim=len_tag+extra_dim
        start_time = time.time()
        (X_new,word_vectors)=embedEncode(sentences,dim,len_token,len(Z),X,sg=sgSign)
        embedTimeS=time.time() - start_time
        sgStr="cbow"
        if sgSign:
           sgStr="skip-gram"
        
        with open("GauSpecBenchmark.txt", "a") as f: 
           f.write(str(extra_dim)+","+str(dim)+","+sgStr+","+str(embedTimeS)+",") 
        Xseq=np.concatenate(X_new, axis=0)
        del X_new
        start_time2 = time.time()
        outDict={}
        P_1_vec=P_matrix_1d_vec(list_P1,dim,word_vectors)
        #del list_P1
        print "P1 success"
        P_ac_vec=P_matrix_2d_vec(list_ac,dim,word_vectors)
        #del list_ac
        _,_,Uct=scipy.sparse.linalg.svds(P_ac_vec,k=len_tag)
        del P_ac_vec
        P_ab=P_matrix_2d_vec(list_ab,dim,word_vectors)
        #del list_ab
        print "Pac success"
        P_abc=P_matrix_3d_vec(list_abc,dim,word_vectors)
        #del list_abc
        print "Tensor success"
        Ua,_,Ubt=scipy.sparse.linalg.svds(P_ab,k=len_tag)
        Ub=np.transpose(Ubt)
        Uc=np.transpose(Uct)
        W=np.linalg.inv(sandwichP(Ua,P_ab,Ub))
        del P_ab
        theta=randMatrixGen(len_tag)
        print "before L success!"
        L,R=constructL(len_tag,P_abc,theta,Uc,W,dim,Ua,Ub)

        del P_abc,W,Ua,Ub
        O=np.dot(np.dot(Uc,np.linalg.inv(theta)),L)

        del L,theta
        T=np.dot(np.linalg.inv(np.dot(np.transpose(Uc),O)),R)
        Tprime,negRatio=negativeHandlingO(T)
        del R,Uc
        print O.shape

        pi=np.dot(np.linalg.pinv(O),P_1_vec)
        print T.shape
        print pi.shape
        matTimeS=time.time() - start_time2
        with open("GauSpecBenchmark.txt", "a") as f: 
            f.write(str(matTimeS)+",") 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start_time3 = time.time()
            q=hmmGaussianSpectral(np.transpose(negativeHandlingArray(pi)),np.transpose(negativeHandling(T)),np.transpose(O),Xseq,Zseq,dim,len_tag,lenInfo)
            covTimeS=time.time() - start_time3
            fullTimeS=covTimeS+matTimeS+embedTimeS
            with open("GauSpecBenchmark.txt", "a") as f: 
                f.write(str(covTimeS)+","+str(fullTimeS)+","+str(negRatio)+","+str(q)+",\n")
