from scipy.stats.stats import pearsonr,spearmanr
import numpy as np
f="MultiEMBenchmark.txt"
f1="MultiEMMean.txt"
f2="MultiEMSTD.txt"
with open(f1,"w") as fileL:
    fileL.write("\n")
with open(f2,"w") as fileL:
    fileL.write("\n")
with open(f, "r") as ins:
     count=0
     modCount=0
     copyCount=0
     sumArray=np.zeros(2)
     totalList=[]
     for line in ins:
         if count==0:
            with open(f1,"a") as fileL:
                 fileL.write(line+"\n")
            with open(f2,"a") as fileL:
                 fileL.write(line+"\n")
         if len(line)<3:
            break

         if count>0:
            w=line.split(",")
            xy=w[1:3]
            xyList=[float(item) for item in xy]
            totalList.append(xyList)
            xyArray=np.asarray(xyList)
            sumArray=xyArray+sumArray
            #print uw[0:2]
            
         if modCount==5:
            sumArray=np.true_divide(sumArray,5)
            sumList=sumArray.tolist()
            strList=[str(item) for item in sumList]
            totalArray=np.array(totalList)
            meanArray=np.mean(totalArray,axis=0)
            stdArray=np.std(totalArray,axis=0)
           
            str1= w[0]+","+(",").join([str(item) for item in meanArray.tolist()])
            str2= w[0]+","+(",").join([str(item) for item in stdArray.tolist()])
            with open(f1,"a") as fileL:
                 fileL.write(str1+"\n")
            with open(f2,"a") as fileL:
                 fileL.write(str2+"\n")
            copyCount=5
         else:
            copyCount=-1
         if copyCount==5:
            modCount=0
            sumArray=np.zeros(2)
            totalList=[]
         count=count+1 
         modCount=modCount+1
     