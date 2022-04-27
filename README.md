# CSC381_FinalProject
This code produces recommendations by implementing 10 different recommender algorithms.

Instructions on how to run each algorithm:

User-based CF recommendations:
1) Read prefered dataset (rml-100k or critics)
2) Call Simu
3) Select similarity method
   Call Write Distance (wd) or Read Distance (rd) for Euclidean Distance
   Call Write Pearson (wp) or Read Pearson (rp) for Pearson Correlation
4) Call RECS, select user to generate recommendations and specify the number of recommendations
5) Call LCVSIM for error tests and measure accuracy of algorithm

Item-based CF recommendations:
1) Read prefered dataset (rml-100k or critics)
2) Call Sim
3) Select Similarity Method
   Call Write Distance (wd) or Read Distance (rd) for Euclidean Distance
   Call Write Pearson (wp) or Read Pearson (rp) for Pearson Correlation
4) Call RECS, select user to generate recommendations and specify the number of recommendations
5) Call LCVSIM for error tests and measure accuracy of algorithm

TFIDF
1) Read prefered dataset
2) Call TFIDF
3) Call RECS, select user to generate recommendations and specify the number of recommendations
4) Call LCVSIM for error tests and measure accuracy of algorithm

MF-ALS
1) Read dataset as a pandas dataframe (pd-rml or pd-critics)
2) Call Test/Train command to train the dataset
3) Call MF-ALS (optimal parameters are already passed in)
4) Call RECS, select user to generate recommendations and specify the number of recommendations

MF-SGD
1) Read dataset as a pandas dataframe (pd-rml or pd-critics)
2) Call Test/Train command to train the dataset
3) Call MF-SGD (optimal parameters are already passed in)
4) Call RECS, select user to generate recommendations and specify the number of recommendations

Hybrid-Based (Item-TFIDF)
1) Read prefered dataset (rml-100k or critics)
2) Call Sim
3) Select Similarity Method
   Call Write Distance (wd) or Read Distance (rd) for Euclidean Distance
   Call Write Pearson (wp) or Read Pearson (rp) for Pearson Correlation
4) Call TFIDF
5) Call H(ybrid) to specify the request for a hybrid approach
6) Call RECS, select user to generate recommendations and specify the number of recommendations
7) Call LCVSIM for error tests and measure accuracy of algorithm

Deep Learning
1) Read preferred dataset
2) Call NCF command
3) Call RECS, select user to generate recommendations and specify the number of recommendations
4) Call LCVSIM for error tests and measure accuracy of algorithm 
