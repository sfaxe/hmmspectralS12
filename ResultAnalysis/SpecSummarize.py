from scipy.stats.stats import pearsonr,spearmanr
import numpy as np
f="GauSpecBenchmark.txt"
f1="GauSpecMean.txt"
f2="GauSpecSTD.txt"
with open(f1,"w") as fileL:
    fileL.write("\n")
with open(f2,"w") as fileL:
    fileL.write("\n")
with open(f, "r") as ins:
     count=0
     negRatio=[]
     vMeasure=[]
     modCount=0
     copyCount=0
     sumArray=np.zeros(6)
     totalList=[]
     for line in ins:
         if count==0:
            with open(f1,"a") as fileL:
                 fileL.write(line+"\n")
            with open(f2,"a") as fileL:
                 fileL.write(line+"\n")

         if count>0:
            w=line.split(",")
            uw=w[-3:]
            firstItem=(",").join(w[0:3])
            xy=w[3:9]
            xyList=[float(item) for item in xy]
            totalList.append(xyList)
            xyArray=np.asarray(xyList)
            sumArray=xyArray+sumArray
            #print uw[0:2]
            negRatio.append(float(uw[0]))
            vMeasure.append(float(uw[1]))
         if modCount==5:
            sumArray=np.true_divide(sumArray,5)
            sumList=sumArray.tolist()
            strList=[str(item) for item in sumList]
            totalArray=np.array(totalList)
            meanArray=np.mean(totalArray,axis=0)
            stdArray=np.std(totalArray,axis=0)
           
            str1= firstItem+","+(",").join([str(item) for item in meanArray.tolist()])
            str2= firstItem+","+(",").join([str(item) for item in stdArray.tolist()])
            with open(f1,"a") as fileL:
                 fileL.write(str1+"\n")
            with open(f2,"a") as fileL:
                 fileL.write(str2+"\n")
            copyCount=5
         else:
            copyCount=-1
         if copyCount==5:
            modCount=0
            sumArray=np.zeros(6)
            totalList=[]
         count=count+1 
         modCount=modCount+1
     a=np.asarray(negRatio)
     b=np.asarray(vMeasure)
     maxNeg=np.amax(negRatio)
     minNeg=np.amin(negRatio)
     pearsonC,pvalPears=pearsonr(a,b)     
     spearmanrC,pvalSpear=spearmanr(a,b)
     print "max NegRatio:",maxNeg
     print "min NegRatio:",minNeg
     print "The pairs of (negative Ratio, Vmeasure):",len(vMeasure)
     print "Pearson Correlation=",pearsonC,"p value=",pvalPears
     print "Spearman Correlation=",spearmanrC,"p value=",pvalSpear