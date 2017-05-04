Package requirement:
Numpy, Scipy, Scikit-learn
Gensim:(https://radimrehurek.com/gensim/)
(First follow instructions on above. 
If it fails on Mac OS, try $pip install --upgrade --ignore-installed gensim)

hmmlearn: pip install https://github.com/hmmlearn/hmmlearn/archive/d9eabf4a6b493c8867a8b14c7fbb6ac8db84fe2b.zip
progressbar: https://pypi.python.org/pypi/progressbar

version quick-check:
gensim==1.0.1
hmmlearn==0.2.1
numpy==1.12.1
progressbar==2.3
scikit-learn==0.18.1
scipy==0.19.0

Go to each subfolder. Each benchmarking will takes a very long time. However, there will be benchmark results continuously written to a text
file (see sampleoutput within each subfolder.
GaussianEM: python GauEM.py
GaussianSpectral: python GauSpec.py
MultinomialEM: python multiEM.py

The generated benchmark text can be analyzed. In ResultAnalysis folder, one can run: python xxxSummarize.py to get mean values and std 
(standard deviation) values.
(xxx is replaced by "Spec","multiEM","GauEM" respectively).
