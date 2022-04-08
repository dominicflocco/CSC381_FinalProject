'''
LOOCV results for critics data using 
Similarity methods: sim_distance(), sim_pearson()
Recommenders: User-based CF, Item-based CF

Examples of Hypothesis Testing
==> Only for use in the CSC 381 Recommender Systems course.

Author: Carlos Seminario

'''

import os
import pickle
import numpy as np
#from matplotlib import pyplot as plt
from scipy import stats # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html#scipy.stats.ttest_ind
'''
scipy.stats.ttest_ind(a, b, axis=0, equal_var=True, nan_policy='propagate')
 Calculate the T-test for the means of two independent samples of scores.
 This is a two-sided test for the null hypothesis that 2 independent samples 
 have identical average (expected) values. This test assumes that the 
 populations have identical variances by default.
'''

def print_loocv_results(sq_diffs_info):
    ''' Print LOOCV SIM results '''

    error_list = []
    for user in sq_diffs_info:
        for item in sq_diffs_info[user]:
            for data_point in sq_diffs_info[user][item]:
                #print ('User: %s, Item: %s, Prediction: %.5f, Actual: %.5f, Error: %.5f' %\
            #      (user, item, data_point[0], data_point[1], data_point[2]))                
                error_list.append(data_point[2]) # save for MSE calc
                
    #print()
    error = sum(error_list)/len(error_list)          
    print ('MSE =', error)
    
    return(error, error_list)
                
                
def main():
    ''' User interface for Python console '''
    
    path = os.getcwd() # this gets the current working directory
                       # you can customize path for your own computer here
    print('\npath: %s' % path) # debug    
    
    print()
    # Load LOOCV results from file, print
    print('Results for sim_distance, user, lcv:')
    sq_diffs_info = pickle.load(open( "save_sq_diffs_info_distance_using_critics_user_lcv.p", "rb" ))
    distance_errors_u_lcv_MSE, distance_errors_u_lcv = print_loocv_results(sq_diffs_info)   
    print()
    # Load LOOCV results from file, print
    print('Results for sim_pearson, user, lcv:')
    sq_diffs_info = pickle.load(open( "save_sq_diffs_info_pearson_using_critics_user_lcv.p", "rb" ))
    pearson_errors_u_lcv_MSE, pearson_errors_u_lcv = print_loocv_results(sq_diffs_info) 
    
    print()
    print ('t-test for User-LCV distance vs pearson',len(distance_errors_u_lcv), len(pearson_errors_u_lcv))
    print ('Null Hypothesis is that the means (MSE values for User-LCV distance and pearson) are equal')
    
    ## Calc with the scipy function
    t_u_lcv, p_u_lcv = stats.ttest_ind(distance_errors_u_lcv,pearson_errors_u_lcv)
    print("t = " + str(t_u_lcv))
    print("p = " + str(p_u_lcv))
    print()
    print('==>> Unable to reject null hypothesis that the means are equal') # The two-tailed p-value    
    print('==>> The means may or may not be equal')
    
    input('\nContinue? ')

    print()
    # Load LOOCV SIM results from file, print
    print('Results for sim_distance, item, lcvsim:')
    sq_diffs_info = pickle.load(open( "save_sq_diffs_info_distance_using_sim_matrix_item_lcvsim.p", "rb" ))
    distance_errors_i_lcvsim_MSE, distance_errors_i_lcvsim = print_loocv_results(sq_diffs_info)   
    print()
    # Load LOOCV SIM results from file, print
    print('Results for sim_pearson, item, lcvsim:')
    sq_diffs_info = pickle.load(open( "save_sq_diffs_info_pearson_using_sim_matrix_item_lcvsim.p", "rb" ))
    pearson_errors_i_lcvsim_MSE, pearson_errors_i_lcvsim = print_loocv_results(sq_diffs_info) 
    
    print()
    print ('t-test for Item-LCVSIM pearson vs distance', len(pearson_errors_i_lcvsim), len(distance_errors_i_lcvsim))
    print ('Null Hypothesis is that the means (MSE values for Item-LCVSIM pearson and distance) are equal')
    
    ## Calc with the scipy function
    t_i_lcvsim, p_i_lcvsim = stats.ttest_ind(pearson_errors_i_lcvsim, distance_errors_i_lcvsim)
    print("t = " + str(t_i_lcvsim))
    print("p = " + str(p_i_lcvsim))
    print('==>> Unable to reject null hypothesis that the means are equal') 
    print('==>> The means may or may not be equal')
    
    input('\nContinue? ')
    
    print()
    print ('Cross t-tests')
    
    print()
    print ('t-test for Item-LCVSIM distance vs User-LCV distance',len(distance_errors_i_lcvsim), len(distance_errors_u_lcv))
    print ('Null Hypothesis is that the means (MSE values for Item-LCVSIM distance and User-LCV distance) are equal')
    
    ## Calc with the scipy function
    t_u_lcv_i_lcvsim_distance, p_u_lcv_i_lcvsim_distance = stats.ttest_ind(distance_errors_i_lcvsim, distance_errors_u_lcv)
    
    print()
    print('distance_errors_i_lcvsim_MSE, distance_errors_u_lcv_MSE:', distance_errors_i_lcvsim_MSE, distance_errors_u_lcv_MSE)
    print("t = " + str(t_u_lcv_i_lcvsim_distance))
    print("p = " + str(p_u_lcv_i_lcvsim_distance), '==>> Unable to reject null hypothesis that the means are equal')
    print('==>> The means may or may not be equal')

    print()
    print ('t-test for Item-LCVSIM pearson vs User-LCV pearson',len(pearson_errors_i_lcvsim), len(pearson_errors_u_lcv))
    print ('Null Hypothesis is that the means (MSE values for Item-LCVSIM pearson and User-LCV pearson) are equal')
    
    ## Cross Checking with the scipy function
    t_u_lcv_i_lcvsim_pearson, p_u_lcv_i_lcvsim_pearson = stats.ttest_ind(pearson_errors_i_lcvsim, pearson_errors_u_lcv)
    print()
    print('pearson_errors_i_lcvsim_MSE, pearson_errors_u_lcv_MSE:', pearson_errors_i_lcvsim_MSE, pearson_errors_u_lcv_MSE)   
    print("t = " + str(t_u_lcv_i_lcvsim_pearson))
    print("p = " + str(p_u_lcv_i_lcvsim_pearson), '==>> Reject null hypothesis that the means are equal')
    print('==>> The means are not equal')
    


if __name__ == '__main__':
    main()
    
'''
Results for sim_distance, user, lcv:
User: Lisa, Item: Lady in the Water, Prediction: 2.84769, Actual: 2.50000, Error: 0.12089
User: Lisa, Item: Snakes on a Plane, Prediction: 3.72438, Actual: 3.50000, Error: 0.05035
User: Lisa, Item: Just My Luck, Prediction: 2.17257, Actual: 3.00000, Error: 0.68463
User: Lisa, Item: Superman Returns, Prediction: 4.05341, Actual: 3.50000, Error: 0.30627
User: Lisa, Item: You, Me and Dupree, Prediction: 2.38249, Actual: 2.50000, Error: 0.01381
User: Lisa, Item: The Night Listener, Prediction: 3.69958, Actual: 3.00000, Error: 0.48941
User: Gene, Item: Lady in the Water, Prediction: 2.79669, Actual: 3.00000, Error: 0.04133
User: Gene, Item: Snakes on a Plane, Prediction: 3.79776, Actual: 3.50000, Error: 0.08866
User: Gene, Item: Just My Luck, Prediction: 2.70552, Actual: 1.50000, Error: 1.45327
User: Gene, Item: Superman Returns, Prediction: 3.95502, Actual: 5.00000, Error: 1.09198
User: Gene, Item: The Night Listener, Prediction: 3.44607, Actual: 3.00000, Error: 0.19897
User: Gene, Item: You, Me and Dupree, Prediction: 2.44300, Actual: 3.50000, Error: 1.11725
User: Michael, Item: Lady in the Water, Prediction: 2.84741, Actual: 2.50000, Error: 0.12070
User: Michael, Item: Snakes on a Plane, Prediction: 3.86762, Actual: 3.00000, Error: 0.75276
User: Michael, Item: Superman Returns, Prediction: 4.07883, Actual: 3.50000, Error: 0.33504
User: Michael, Item: The Night Listener, Prediction: 3.36213, Actual: 4.00000, Error: 0.40688
User: Claudia, Item: Snakes on a Plane, Prediction: 3.68706, Actual: 3.50000, Error: 0.03499
User: Claudia, Item: Just My Luck, Prediction: 2.21251, Actual: 3.00000, Error: 0.62014
User: Claudia, Item: The Night Listener, Prediction: 3.24753, Actual: 4.50000, Error: 1.56868
User: Claudia, Item: Superman Returns, Prediction: 3.92011, Actual: 4.00000, Error: 0.00638
User: Claudia, Item: You, Me and Dupree, Prediction: 2.35827, Actual: 2.50000, Error: 0.02009
User: Mick, Item: Lady in the Water, Prediction: 2.70215, Actual: 3.00000, Error: 0.08872
User: Mick, Item: Snakes on a Plane, Prediction: 3.65230, Actual: 4.00000, Error: 0.12089
User: Mick, Item: Just My Luck, Prediction: 2.62345, Actual: 2.00000, Error: 0.38869
User: Mick, Item: Superman Returns, Prediction: 4.14918, Actual: 3.00000, Error: 1.32062
User: Mick, Item: The Night Listener, Prediction: 3.56791, Actual: 3.00000, Error: 0.32252
User: Mick, Item: You, Me and Dupree, Prediction: 2.47096, Actual: 2.00000, Error: 0.22181
User: Jack, Item: Lady in the Water, Prediction: 2.79262, Actual: 3.00000, Error: 0.04301
User: Jack, Item: Snakes on a Plane, Prediction: 3.59284, Actual: 4.00000, Error: 0.16578
User: Jack, Item: The Night Listener, Prediction: 3.46458, Actual: 3.00000, Error: 0.21584
User: Jack, Item: Superman Returns, Prediction: 3.93792, Actual: 5.00000, Error: 1.12802
User: Jack, Item: You, Me and Dupree, Prediction: 2.40522, Actual: 3.50000, Error: 1.19855
User: Toby, Item: Snakes on a Plane, Prediction: 3.50381, Actual: 4.50000, Error: 0.99240
User: Toby, Item: You, Me and Dupree, Prediction: 2.77902, Actual: 1.00000, Error: 3.16490
User: Toby, Item: Superman Returns, Prediction: 3.86064, Actual: 4.00000, Error: 0.01942

MSE = 0.5403902654348717

Results for sim_pearson, user, lcv:
User: Lisa, Item: Lady in the Water, Prediction: 3.00000, Actual: 2.50000, Error: 0.25000
User: Lisa, Item: Snakes on a Plane, Prediction: 3.81964, Actual: 3.50000, Error: 0.10217
User: Lisa, Item: Just My Luck, Prediction: 2.13661, Actual: 3.00000, Error: 0.74544
User: Lisa, Item: Superman Returns, Prediction: 3.94484, Actual: 3.50000, Error: 0.19788
User: Lisa, Item: You, Me and Dupree, Prediction: 3.14556, Actual: 2.50000, Error: 0.41675
User: Lisa, Item: The Night Listener, Prediction: 3.62498, Actual: 3.00000, Error: 0.39060
User: Gene, Item: Lady in the Water, Prediction: 2.88701, Actual: 3.00000, Error: 0.01277
User: Gene, Item: Snakes on a Plane, Prediction: 3.98553, Actual: 3.50000, Error: 0.23574
User: Gene, Item: Just My Luck, Prediction: 3.00000, Actual: 1.50000, Error: 2.25000
User: Gene, Item: Superman Returns, Prediction: 4.26516, Actual: 5.00000, Error: 0.53999
User: Gene, Item: The Night Listener, Prediction: 3.55174, Actual: 3.00000, Error: 0.30441
User: Gene, Item: You, Me and Dupree, Prediction: 2.76169, Actual: 3.50000, Error: 0.54511
User: Michael, Item: Snakes on a Plane, Prediction: 3.54649, Actual: 3.00000, Error: 0.29865
User: Michael, Item: Superman Returns, Prediction: 3.87670, Actual: 3.50000, Error: 0.14190
User: Michael, Item: The Night Listener, Prediction: 3.39197, Actual: 4.00000, Error: 0.36970
User: Claudia, Item: Snakes on a Plane, Prediction: 3.75001, Actual: 3.50000, Error: 0.06250
User: Claudia, Item: Just My Luck, Prediction: 2.48623, Actual: 3.00000, Error: 0.26396
User: Claudia, Item: The Night Listener, Prediction: 3.24207, Actual: 4.50000, Error: 1.58238
User: Claudia, Item: Superman Returns, Prediction: 3.60773, Actual: 4.00000, Error: 0.15388
User: Claudia, Item: You, Me and Dupree, Prediction: 2.94384, Actual: 2.50000, Error: 0.19699
User: Mick, Item: Lady in the Water, Prediction: 2.72969, Actual: 3.00000, Error: 0.07307
User: Mick, Item: Snakes on a Plane, Prediction: 3.86336, Actual: 4.00000, Error: 0.01867
User: Mick, Item: Just My Luck, Prediction: 3.00000, Actual: 2.00000, Error: 1.00000
User: Mick, Item: Superman Returns, Prediction: 4.19565, Actual: 3.00000, Error: 1.42957
User: Mick, Item: The Night Listener, Prediction: 3.52144, Actual: 3.00000, Error: 0.27190
User: Mick, Item: You, Me and Dupree, Prediction: 2.19209, Actual: 2.00000, Error: 0.03690
User: Jack, Item: Lady in the Water, Prediction: 2.82184, Actual: 3.00000, Error: 0.03174
User: Jack, Item: Snakes on a Plane, Prediction: 3.80658, Actual: 4.00000, Error: 0.03741
User: Jack, Item: The Night Listener, Prediction: 3.60264, Actual: 3.00000, Error: 0.36318
User: Jack, Item: Superman Returns, Prediction: 4.05390, Actual: 5.00000, Error: 0.89510
User: Jack, Item: You, Me and Dupree, Prediction: 2.94873, Actual: 3.50000, Error: 0.30390
User: Toby, Item: Snakes on a Plane, Prediction: 3.70000, Actual: 4.50000, Error: 0.64000
User: Toby, Item: You, Me and Dupree, Prediction: 2.00000, Actual: 1.00000, Error: 1.00000
User: Toby, Item: Superman Returns, Prediction: 3.87500, Actual: 4.00000, Error: 0.01562

MSE = 0.44640825000017764


Results for sim_distance:
User: Lisa, Item: Lady in the Water, Prediction: 3.03921, Actual: 2.50000, Error: 0.29074
User: Lisa, Item: Snakes on a Plane, Prediction: 2.91983, Actual: 3.50000, Error: 0.33659
User: Lisa, Item: Just My Luck, Prediction: 2.92826, Actual: 3.00000, Error: 0.00515
User: Lisa, Item: Superman Returns, Prediction: 2.94880, Actual: 3.50000, Error: 0.30382
User: Lisa, Item: You, Me and Dupree, Prediction: 2.97611, Actual: 2.50000, Error: 0.22668
User: Lisa, Item: The Night Listener, Prediction: 2.96502, Actual: 3.00000, Error: 0.00122
User: Gene, Item: Lady in the Water, Prediction: 3.20121, Actual: 3.00000, Error: 0.04049
User: Gene, Item: Snakes on a Plane, Prediction: 3.23158, Actual: 3.50000, Error: 0.07205
User: Gene, Item: Just My Luck, Prediction: 3.49186, Actual: 1.50000, Error: 3.96751
User: Gene, Item: Superman Returns, Prediction: 2.94877, Actual: 5.00000, Error: 4.20755
User: Gene, Item: The Night Listener, Prediction: 3.23443, Actual: 3.00000, Error: 0.05496
User: Gene, Item: You, Me and Dupree, Prediction: 2.99820, Actual: 3.50000, Error: 0.25180
User: Michael, Item: Lady in the Water, Prediction: 3.52003, Actual: 2.50000, Error: 1.04046
User: Michael, Item: Snakes on a Plane, Prediction: 3.30757, Actual: 3.00000, Error: 0.09460
User: Michael, Item: Superman Returns, Prediction: 3.16526, Actual: 3.50000, Error: 0.11205
User: Michael, Item: The Night Listener, Prediction: 2.92984, Actual: 4.00000, Error: 1.14525
User: Claudia, Item: Snakes on a Plane, Prediction: 3.64770, Actual: 3.50000, Error: 0.02182
User: Claudia, Item: Just My Luck, Prediction: 3.57625, Actual: 3.00000, Error: 0.33206
User: Claudia, Item: The Night Listener, Prediction: 3.22781, Actual: 4.50000, Error: 1.61846
User: Claudia, Item: Superman Returns, Prediction: 3.45510, Actual: 4.00000, Error: 0.29692
User: Claudia, Item: You, Me and Dupree, Prediction: 3.73115, Actual: 2.50000, Error: 1.51572
User: Mick, Item: Lady in the Water, Prediction: 2.74660, Actual: 3.00000, Error: 0.06421
User: Mick, Item: Snakes on a Plane, Prediction: 2.68769, Actual: 4.00000, Error: 1.72217
User: Mick, Item: Just My Luck, Prediction: 2.95459, Actual: 2.00000, Error: 0.91125
User: Mick, Item: Superman Returns, Prediction: 2.92444, Actual: 3.00000, Error: 0.00571
User: Mick, Item: The Night Listener, Prediction: 2.82438, Actual: 3.00000, Error: 0.03084
User: Mick, Item: You, Me and Dupree, Prediction: 2.90881, Actual: 2.00000, Error: 0.82593
User: Jack, Item: Lady in the Water, Prediction: 3.73910, Actual: 3.00000, Error: 0.54626
User: Jack, Item: Snakes on a Plane, Prediction: 3.61075, Actual: 4.00000, Error: 0.15152
User: Jack, Item: The Night Listener, Prediction: 3.77531, Actual: 3.00000, Error: 0.60111
User: Jack, Item: Superman Returns, Prediction: 3.40748, Actual: 5.00000, Error: 2.53613
User: Jack, Item: You, Me and Dupree, Prediction: 3.50904, Actual: 3.50000, Error: 0.00008
User: Toby, Item: Snakes on a Plane, Prediction: 2.86284, Actual: 4.50000, Error: 2.68030
User: Toby, Item: You, Me and Dupree, Prediction: 4.24791, Actual: 1.00000, Error: 10.54889
User: Toby, Item: Superman Returns, Prediction: 3.15948, Actual: 4.00000, Error: 0.70647

MSE = 1.0647645675666437

Results for sim_pearson:
User: Lisa, Item: Lady in the Water, Prediction: 3.28970, Actual: 2.50000, Error: 0.62363
User: Lisa, Item: Snakes on a Plane, Prediction: 2.62769, Actual: 3.50000, Error: 0.76092
User: Lisa, Item: Just My Luck, Prediction: 3.00000, Actual: 3.00000, Error: 0.00000
User: Lisa, Item: Superman Returns, Prediction: 2.58889, Actual: 3.50000, Error: 0.83011
User: Lisa, Item: You, Me and Dupree, Prediction: 3.16374, Actual: 2.50000, Error: 0.44055
User: Lisa, Item: The Night Listener, Prediction: 3.00000, Actual: 3.00000, Error: 0.00000
User: Gene, Item: Lady in the Water, Prediction: 3.96177, Actual: 3.00000, Error: 0.92500
User: Gene, Item: Snakes on a Plane, Prediction: 3.25539, Actual: 3.50000, Error: 0.05984
User: Gene, Item: Just My Luck, Prediction: 3.00000, Actual: 1.50000, Error: 2.25000
User: Gene, Item: Superman Returns, Prediction: 3.30602, Actual: 5.00000, Error: 2.86958
User: Gene, Item: The Night Listener, Prediction: 1.50000, Actual: 3.00000, Error: 2.25000
User: Gene, Item: You, Me and Dupree, Prediction: 4.32747, Actual: 3.50000, Error: 0.68471
User: Michael, Item: Lady in the Water, Prediction: 3.19491, Actual: 2.50000, Error: 0.48290
User: Michael, Item: Snakes on a Plane, Prediction: 2.62769, Actual: 3.00000, Error: 0.13861
User: Michael, Item: Superman Returns, Prediction: 2.59321, Actual: 3.50000, Error: 0.82227
User: Claudia, Item: Snakes on a Plane, Prediction: 4.00000, Actual: 3.50000, Error: 0.25000
User: Claudia, Item: Just My Luck, Prediction: 4.50000, Actual: 3.00000, Error: 2.25000
User: Claudia, Item: The Night Listener, Prediction: 3.00000, Actual: 4.50000, Error: 2.25000
User: Claudia, Item: Superman Returns, Prediction: 2.64525, Actual: 4.00000, Error: 1.83536
User: Claudia, Item: You, Me and Dupree, Prediction: 4.00000, Actual: 2.50000, Error: 2.25000
User: Mick, Item: Lady in the Water, Prediction: 3.27156, Actual: 3.00000, Error: 0.07374
User: Mick, Item: Snakes on a Plane, Prediction: 3.00000, Actual: 4.00000, Error: 1.00000
User: Mick, Item: Just My Luck, Prediction: 3.00000, Actual: 2.00000, Error: 1.00000
User: Mick, Item: Superman Returns, Prediction: 2.56576, Actual: 3.00000, Error: 0.18857
User: Mick, Item: The Night Listener, Prediction: 2.00000, Actual: 3.00000, Error: 1.00000
User: Mick, Item: You, Me and Dupree, Prediction: 3.00000, Actual: 2.00000, Error: 1.00000
User: Jack, Item: Lady in the Water, Prediction: 4.20270, Actual: 3.00000, Error: 1.44648
User: Jack, Item: Snakes on a Plane, Prediction: 3.25539, Actual: 4.00000, Error: 0.55445
User: Jack, Item: Superman Returns, Prediction: 3.35046, Actual: 5.00000, Error: 2.72097
User: Jack, Item: You, Me and Dupree, Prediction: 4.32747, Actual: 3.50000, Error: 0.68471
User: Toby, Item: Snakes on a Plane, Prediction: 4.00000, Actual: 4.50000, Error: 0.25000
User: Toby, Item: You, Me and Dupree, Prediction: 4.00000, Actual: 1.00000, Error: 9.00000
User: Toby, Item: Superman Returns, Prediction: 1.50836, Actual: 4.00000, Error: 6.20828

MSE = 1.4272933418206895
'''
