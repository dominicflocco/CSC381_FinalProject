'''
CSC381: Building a simple Recommender System

The final code package is a collaborative programming effort between the
CSC381 student(s) named below, the class instructor (Carlos Seminario), and
source code from Programming Collective Intelligence, Segaran 2007.
This code is for academic use/purposes only.

CSC381 Programmer/Researcher: Dominic Flocco

'''

import os
import numpy as np
import matplotlib.pyplot as plt
from math import *
import math
import copy
import pickle
import pandas as pd
import timeit
from scipy.stats import spearmanr
from scipy.stats import kendalltau


def from_file_to_dict(path, datafile, itemfile):
    ''' Load user-item matrix from specified file 
        
        Parameters:
        -- path: directory path to datafile and itemfile
        -- datafile: delimited file containing userid, itemid, rating
        -- itemfile: delimited file that maps itemid to item name
        
        Returns:
        -- prefs: a nested dictionary containing item ratings for each user
    
    '''
    
    # Get movie titles, place into movies dictionary indexed by itemID
    movies={}
    try:
        with open (path + '/' + itemfile, encoding='iso8859') as myfile: 
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (id,title)=line.split('|')[0:2]
                movies[id]=title.strip()
    
    # Error processing
    except UnicodeDecodeError as ex:
        print (ex)
        print (len(movies), line, id, title)
        return {}
    except Exception as ex:
        print (ex)
        print (len(movies))
        return {}
    
    # Load data into a nested dictionary
    prefs={}
    for line in open(path+'/'+ datafile, encoding='iso8859'):
        #print(line, line.split('\t')) #debug
        (user,movieid,rating,ts)=line.split('\t')
        user = user.strip() # remove spaces
        movieid = movieid.strip() # remove spaces
        prefs.setdefault(user,{}) # make it a nested dicitonary
        prefs[user][movies[movieid]]=float(rating)
    
    #return a dictionary of preferences
    return prefs

def data_stats(prefs, filename):
    ''' Computes/prints descriptive analytics:
        -- Total number of users, items, ratings
        -- Overall average rating, standard dev (all users, all items)
        -- Average item rating, standard dev (all users)
        -- Average user rating, standard dev (all items)
        -- Matrix ratings sparsity
        -- Ratings distribution histogram (all users, all items)

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- filename: string containing name of file being analyzed
        
        Returns:
        -- None

    '''
    num_users = len(prefs)
    
    ratings = []
    item_ratings = {}
    
    for user in prefs.keys(): 
        user_ratings = prefs[user]
        for item in user_ratings.keys():
            if item in item_ratings:
                item_ratings[item].append(user_ratings[item])
            else: 
                item_ratings[item] = [user_ratings[item]]
            ratings.append(user_ratings[item])
        
    num_items = len(item_ratings)
    num_ratings = len(ratings)
    
    overall_avg_rating = sum(ratings)/num_ratings
    overall_std = np.std(ratings)

    item_avgs = {}
    for item in item_ratings.keys():
        item_avgs[item] = sum(item_ratings[item])/len(item_ratings[item])
        
    item_avg_rating = sum([item_avgs[item] for item in item_avgs.keys()])/num_items
    item_std = np.std([item_avgs[item] for item in item_avgs.keys()])

    user_avgs = {}
    user_num_ratings = {}
    for user in prefs.keys():
        user_num_ratings[user] = len([prefs[user][item] for item in prefs[user]])
        user_avgs[user] = sum([prefs[user][item] for item in prefs[user]])/len(prefs[user])

    num_ratings_list = [user_num_ratings[user] for user in prefs.keys()]
    avg_num_user_ratings = np.mean(num_ratings_list)
    std_dev_user_ratings = np.std(num_ratings_list)
    min_user_ratings = np.min(num_ratings_list)
    max_user_ratings = np.max(num_ratings_list) 
    median_user_ratings = np.median(num_ratings_list)

    user_avg_rating = sum([user_avgs[user] for user in user_avgs.keys()])/num_users
    user_std = np.std([user_avgs[user] for user in user_avgs.keys()])

    sparsity = (1 - (num_ratings/(num_users*num_items))) * 100

    print("Number of users: %d" % num_users)
    print("Number of items: %d" % num_items)
    print("Number of ratings: %d"% num_ratings)
    print("Average number of ratings per user: %.2f" % avg_num_user_ratings)
    print("Standard dev of ratings per user: %.2f" % std_dev_user_ratings)
    print("Min number of ratings per user: %.2f" % min_user_ratings)
    print("Max number of ratings per user: %.2f" % max_user_ratings)
    print("Median number of ratings per user: %.2f" % median_user_ratings)
    print("Overall average rating: %.2f out of 5, and std dev of %.2f" % (overall_avg_rating, overall_std))
    # print("Item Avg Ratings List: ", [item_avgs[item] for item in item_avgs.keys()])
    print("Average item rating: %.2f out of 5, and std dev of %.2f" % (item_avg_rating, item_std))
    # print("User Ratings List: ", [user_avgs[user] for user in prefs.keys()])
    print("Average user rating: %.2f out of 5, and std dev of %.2f" % (user_avg_rating, user_std))
    print("User-Item Matrix Sparsity: %.2f" % sparsity, '%')


    plt.hist(ratings, 4, facecolor='blue', alpha = 0.75)
    plt.title("Ratings Histogram")
    #plt.xticks(np.arange(1,6, step=1))
    #plt.yticks(np.arange(0, 19, step=2))
    plt.xlabel('Rating')
    plt.ylabel('Number of user ratings')
    plt.grid()
    plt.show()
    plt.close()
          
def popular_items(prefs, filename): 
    ''' Computes/prints popular items analytics    
        -- popular items: most rated (sorted by # ratings)
        -- popular items: highest rated (sorted by avg rating)
        -- popular items: highest rated items that have at least a 
                          "threshold" number of ratings
        
        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- filename: string containing name of file being analyzed
        
        Returns:
        -- None

    '''
    items_dict = {}
    
    for i in prefs:
        for j in prefs[i].keys():
            items_dict[j] = [0,0,0]
            
    for i in prefs:
        for j in prefs[i].keys():
            items_dict[j][0] += prefs[i][j]
            items_dict[j][1] += 1
    
    for i in items_dict:
        items_dict[i][2] = items_dict[i][0] / items_dict[i][1]
    
    

    items_df_1 = pd.DataFrame(items_dict)
    items_df = items_df_1.transpose()
    
    ratings_sorted_items_df = items_df.sort_values(by=[1], ascending=False)
    
    print('\nPopular items -- most rated: \nTitle \t\t\t\t #Ratings \t Avg Rating \n')
    print('%s \t\t %d \t\t %.2f' % (ratings_sorted_items_df.index[0],ratings_sorted_items_df.iloc[0][1], ratings_sorted_items_df.iloc[0][2]))
    print('%s \t\t\t %d \t\t %.2f' % (ratings_sorted_items_df.index[1],ratings_sorted_items_df.iloc[1][1], ratings_sorted_items_df.iloc[1][2]))
    print('%s \t\t\t %d \t\t %.2f' % (ratings_sorted_items_df.index[2],ratings_sorted_items_df.iloc[2][1], ratings_sorted_items_df.iloc[2][2]))
    print('%s \t %d \t\t %.2f' % (ratings_sorted_items_df.index[3],ratings_sorted_items_df.iloc[3][1], ratings_sorted_items_df.iloc[3][2]))
    print('%s \t\t %d \t\t %.2f' % (ratings_sorted_items_df.index[4],ratings_sorted_items_df.iloc[4][1], ratings_sorted_items_df.iloc[4][2]))
    
    avg_ratings_sorted_items_df = items_df.sort_values(by=[2], ascending=False)
    
    print('\nPopular items -- highest rated: \nTitle \t\t\t\t Avg Rating \t #Ratings \n')
    print('%s \t %.2f \t\t %d' % (avg_ratings_sorted_items_df.index[0],avg_ratings_sorted_items_df.iloc[0][2], avg_ratings_sorted_items_df.iloc[0][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df.index[1],avg_ratings_sorted_items_df.iloc[1][2], avg_ratings_sorted_items_df.iloc[1][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df.index[2],avg_ratings_sorted_items_df.iloc[2][2], avg_ratings_sorted_items_df.iloc[2][1]))
    print('%s \t %.2f \t\t %d' % (avg_ratings_sorted_items_df.index[3],avg_ratings_sorted_items_df.iloc[3][2], avg_ratings_sorted_items_df.iloc[3][1]))
    print('%s \t %.2f \t\t %d' % (avg_ratings_sorted_items_df.index[4],avg_ratings_sorted_items_df.iloc[4][2], avg_ratings_sorted_items_df.iloc[4][1]))
    
    items_df_min = items_df[items_df[1] >= 20]
    avg_ratings_sorted_items_df_min = items_df_min.sort_values(by=[2], ascending=False)
    
    print('\nOverall best rated items (number of ratings >= 20): \nTitle \t\t\t\t Avg Rating \t #Ratings \n')
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[0],avg_ratings_sorted_items_df_min.iloc[0][2], avg_ratings_sorted_items_df_min.iloc[0][1]))
    print('%s \t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[1],avg_ratings_sorted_items_df_min.iloc[1][2], avg_ratings_sorted_items_df_min.iloc[1][1]))
    print('%s \t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[2],avg_ratings_sorted_items_df_min.iloc[2][2], avg_ratings_sorted_items_df_min.iloc[2][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[3],avg_ratings_sorted_items_df_min.iloc[3][2], avg_ratings_sorted_items_df_min.iloc[3][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[4],avg_ratings_sorted_items_df_min.iloc[4][2], avg_ratings_sorted_items_df_min.iloc[4][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[5],avg_ratings_sorted_items_df_min.iloc[5][2], avg_ratings_sorted_items_df_min.iloc[5][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[6],avg_ratings_sorted_items_df_min.iloc[6][2], avg_ratings_sorted_items_df_min.iloc[6][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[7],avg_ratings_sorted_items_df_min.iloc[7][2], avg_ratings_sorted_items_df_min.iloc[7][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[8],avg_ratings_sorted_items_df_min.iloc[8][2], avg_ratings_sorted_items_df_min.iloc[8][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[9],avg_ratings_sorted_items_df_min.iloc[9][2], avg_ratings_sorted_items_df_min.iloc[9][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[10],avg_ratings_sorted_items_df_min.iloc[10][2], avg_ratings_sorted_items_df_min.iloc[10][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[11],avg_ratings_sorted_items_df_min.iloc[11][2], avg_ratings_sorted_items_df_min.iloc[11][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[12],avg_ratings_sorted_items_df_min.iloc[12][2], avg_ratings_sorted_items_df_min.iloc[12][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[13],avg_ratings_sorted_items_df_min.iloc[13][2], avg_ratings_sorted_items_df_min.iloc[13][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[14],avg_ratings_sorted_items_df_min.iloc[14][2], avg_ratings_sorted_items_df_min.iloc[14][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[15],avg_ratings_sorted_items_df_min.iloc[15][2], avg_ratings_sorted_items_df_min.iloc[15][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[16],avg_ratings_sorted_items_df_min.iloc[16][2], avg_ratings_sorted_items_df_min.iloc[16][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[17],avg_ratings_sorted_items_df_min.iloc[17][2], avg_ratings_sorted_items_df_min.iloc[17][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[18],avg_ratings_sorted_items_df_min.iloc[18][2], avg_ratings_sorted_items_df_min.iloc[18][1]))
    print('%s \t\t %.2f \t\t %d' % (avg_ratings_sorted_items_df_min.index[19],avg_ratings_sorted_items_df_min.iloc[19][2], avg_ratings_sorted_items_df_min.iloc[19][1]))

def sim_pearson(prefs,p1,p2, weight):
    '''
        Calculate Pearson Correlation similarity 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2
        
        Returns:
        -- Pearson Correlation similarity as a float
        
    Source: Programming Collective Intelligence, Segaran 2007
    '''
    # Get the list of mutually rated items
    si={}
    for item in prefs[p1]: 
        if item in prefs[p2]: 
            si[item]=1
  
    # if there are no ratings in common, return 0
    if len(si)==0: 
        return 0
  
    # Sum calculations
    n=len(si)
    
    
    # Sums of all the preferences
    sum1=sum([prefs[p1][it] for it in si])
    sum2=sum([prefs[p2][it] for it in si])
    
    # Sums of the squares
    sum1Sq=sum([pow(prefs[p1][it],2) for it in si])
    sum2Sq=sum([pow(prefs[p2][it],2) for it in si])   
    
    # Sum of the products
    pSum=sum([prefs[p1][it]*prefs[p2][it] for it in si])
    
    # Calculate r (Pearson score)
    num=pSum-(sum1*sum2/n)
    den=sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
    if den==0: 
        return 0
    
    if n < weight and weight != 1:
        r=(num/den)*(n/weight)
    else: 
        r = (num/den)
    
  
    return r

def sim_distance(prefs,person1,person2, weight):
    '''
        Calculate Euclidean distance similarity 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2
        
        Returns:
        -- Euclidean distance similarity as a float
    
    Source: Programming Collective Intelligence, Segaran 2007
    '''
    
    # Get the list of shared_items
    si={}
    for item in prefs[person1]: 
        if item in prefs[person2]: 
            si[item]=1
    
    # if they have no ratings in common, return 0
    if len(si)==0: 
        return 0
    
    n = len(si)
    
    # Add up the squares of all the differences
    sum_of_squares=sum([pow(prefs[person1][item]-prefs[person2][item],2) 
                        for item in prefs[person1] if item in prefs[person2]])
    
    '''
    ## FYI, This is what the list comprehension above breaks down to ..
    ##
    sum_of_squares = 0
    for item in prefs[person1]:
        if item in prefs[person2]:
            #print(item, prefs[person1][item], prefs[person2][item])
            sq = pow(prefs[person1][item]-prefs[person2][item],2)
            #print (sq)
            sum_of_squares += sq
    '''
    
    sim = 1/(1+sqrt(sum_of_squares)) 
    if n < weight and weight != 1:
        sim = sim * (n/weight)

    return sim
    
def sim_tanimoto(prefs,p1,p2,weight):
    '''
    Returns the Tanimoto correlation coefficient for vectors p1 and p2 
    https://en.wikipedia.org/wiki/Jaccard_index#Tanimoto_similarity_and_distance
    
    '''
    
    # Get the list of mutually rated items
    si={}
    for item in prefs[p1]: 
        if item in prefs[p2]: 
            si[item]=1
  
    # if there are no ratings in common, return 0
    if len(si)==0: 
        return 0
    n = len(si)
    # Sum calculations
    n=len(si)
    
    # Sums of the squares
    sum1Sq=sum([pow(prefs[p1][it],2) for it in si])
    sum2Sq=sum([pow(prefs[p2][it],2) for it in si])
  
    # Sum of the products
    pSum=sum([prefs[p1][it]*prefs[p2][it] for it in si])
    
    # Calculate r (Tanimoto score)
    num=pSum
    den=sum1Sq + sum2Sq - pSum
    if den==0: 
        return 0
    if n < weight and weight != 1:
        r = (num/den) * (n/weight)
    else:
        r = (num/den)
    return r

def sim_jaccard(prefs, p1, p2, weight):
    
    '''
    The Jaccard similarity index (sometimes called the Jaccard similarity coefficient)
    compares members for two sets to see which members are shared and which are distinct.
    It’s a measure of similarity for the two sets of data, with a range from 0% to 100%. 
    The higher the percentage, the more similar the two populations. Although it’s easy to
    interpret, it is extremely sensitive to small samples sizes and may give erroneous
    results, especially with very small samples or data sets with missing observations.
    https://www.statisticshowto.datasciencecentral.com/jaccard-index/
    https://en.wikipedia.org/wiki/Jaccard_index
    
    The formula to find the Index is:
    Jaccard Index = (the number in both sets) / (the number in either set) * 100
    
    In Steps, that’s:
    Count the number of members which are shared between both sets.
    Count the total number of members in both sets (shared and un-shared).
    Divide the number of shared members (1) by the total number of members (2).
    Multiply the number you found in (3) by 100.
    
    A simple example using set notation: How similar are these two sets?

    A = {0,1,2,5,6}
    B = {0,2,3,4,5,7,9}

    Solution: J(A,B) = |A∩B| / |A∪B| = |{0,2,5}| / |{0,1,2,3,4,5,6,7,9}| = 3/9 = 0.33.

    Notes:
    The cardinality of A, denoted |A| is a count of the number of elements in set A.
    Although it’s customary to leave the answer in decimal form if you’re using set 
    notation, you could multiply by 100 to get a similarity of 33.33%.

    '''
    # Get the lists of mutually rated and unique items
    common_items={}
    unique_items={}
    for item in prefs[p1]: 
        if item in prefs[p2]: 
            # common_items[item]=1 # Case 0: count as common_items if item is rated in both lists 
            if prefs[p1][item] == prefs[p2][item]: # Case 1: rating must match exactly!
            #if abs(prefs[p1][item] - prefs[p2][item]) <= 0.5: # Case 2: rating must be +/- 0.5!
                common_items[item]=1
            else:
                unique_items[item]=1
        else:
            unique_items[item]=1
  
    # if there are no ratings in common, return 0
    if len(common_items)==0: 
        return 0
    n = len(common_items)
    # Sum calculations
    num=len(common_items)
  
    # Calculate Jaccard index
    den=len(common_items) + len(unique_items)
    if den==0: 
        return 0
    if n < weight and weight != 1:
        jaccard_index=(num/den)*(n/weight)
    else: 
        jaccard_index=(num/den)
    return jaccard_index    

def sim_cosine(prefs, p1, p2, weight):
    '''
    Info:
    https://www.geeksforgeeks.org/cosine-similarity/
    https://en.wikipedia.org/wiki/Cosine_similarity
    
    Source for some of the code: https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists

    '''
    # Get the list of mutually rated items
    si={}
    for item in prefs[p1]: 
        if item in prefs[p2]: 
            si[item]=1

    # if there are no ratings in common, return 0
    if len(si)==0: 
        return 0 
    n = len(si)
    # Sum of the products
    sumxy=([prefs[p1][it]*prefs[p2][it] for it in si])
    sumxx=([prefs[p1][it]*prefs[p1][it] for it in si])
    sumyy=([prefs[p2][it]*prefs[p2][it] for it in si])
    #print (sumxy, sumxx, sumyy)
    sumxy=sum(sumxy)
    sumxx=sum(sumxx)
    sumyy=sum(sumyy)
    
    # Calculate r (cosine sim score)
    num = sumxy
    den = sqrt(sumxx*sumyy) 
    
    if den==0: 
        return 0
    if n < weight and weight != 1:
        r=(num/den)*(n/weight)
    else: 
        r = (num/den)
    return r

def sim_spearman(prefs, p1, p2, weight):
    '''
    Calc Spearman's correlation coefficient using scipy function
    
    Enter >>> help(spearmanr) # to get helpful info
    '''
  
    # Get the list of mutually rated items
    si={}
    for item in prefs[p1]: 
        if item in prefs[p2]: 
            si[item]=1
  
    # if there are no ratings in common, return 0
    if len(si)==0: 
        return 0
    
    # Sum calculations
    n=len(si)
    
    # Sums of all the preferences
    data1 = [prefs[p1][it] for it in si]
    data2 = [prefs[p2][it] for it in si]
    
    len1 = len(data1)
    len2 = len(data2)    
    
    coef, p = spearmanr(data1, data2)
    #print('Spearmans correlation coefficient: %.3f' % coef)
    
    if str(coef) == 'nan':
        return 0
    
    # interpret the significance
    '''
    alpha = 0.05
    if p > alpha:
        print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    else:
        print('Samples are correlated (reject H0) p=%.3f' % p)   
    
    '''
    if n < weight and weight != 1:
        coef = coef*(n/weight)

    return coef

def sim_kendall_tau(prefs, p1, p2, weight):
    '''
    Calc Kendall Tau correlation coefficient using scipy function
    
    Enter >>> help(kendalltau) # to get helpful info
    '''
    
    # Get the list of mutually rated items
    si={}
    for item in prefs[p1]: 
        if item in prefs[p2]: 
            si[item]=1
  
    # if there are no ratings in common, return 0
    if len(si)==0: 
        return 0
    n = len(si)
    # Sum calculations
    n=len(si)
    
    # Sums of all the preferences
    data1 = [prefs[p1][it] for it in si]
    data2 = [prefs[p2][it] for it in si]
    
    len1 = len(data1)
    len2 = len(data2)
    
    
    coef, p = kendalltau(data1, data2)
    
    if -1 <= coef <= 1:
        pass
    else:
        coef = 0
        #print(coef, p1, p2)
    
    #sum_coef = 0
    #for it in si:
        #coef, p = kendalltau(prefs[p1][it], prefs[p2][it])
        #coef, p = kendalltau(p1, p2)
        #sum_coef += coef
        #print('Kendall correlation coefficient: %.3f' % coef)
    #coef = sum_coef/n
    
    if n < weight:
        coef = coef *(n/weight)
    '''
    # interpret the significance
    alpha = 0.05
    if p > alpha:
        print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    else:
        print('Samples are correlated (reject H0) p=%.3f' % p)    
    '''
    return coef

def getRecommendations(prefs,person, similarity=sim_pearson):
    '''
        Calculates recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''
    
    totals={}
    simSums={}
    for other in prefs:
      # don't compare me to myself
        if other == person: 
            continue
        sim = similarity(prefs,person,other)
    
        # ignore scores of zero or lower
        if sim <= 0: continue
        for item in prefs[other]:
            
            # only score movies I haven't seen yet
            if item not in prefs[person] or prefs[person][item]==0:
                # Similarity * Score
                totals.setdefault(item,0)
                totals[item]+=prefs[other][item]*sim
                # Sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+=sim
  
    # Create the normalized list
    rankings=[(total/simSums[item],item) for item,total in totals.items()]
  
    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings

def getRecommendationsSim(prefs, person, usersim):
    '''
        Calculates recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- usersim: previously calculated (within simu) user-user similarity matrix
        
        Returns:
        -- A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''
    totals={}
    simSums={}
    for other in prefs:
      # don't compare me to myself
        if other == person: 
            continue
        
        #finds the similarity weight from the user-user similarity matrix that has already been calculated
        # previously -- sim = similarity(prefs,person,other)
        sim = 0
        for tupl in usersim[person]:
            if tupl[1] == other:
                sim = tupl[0]
    
        # ignore scores of zero or lower
        if sim <= 0: continue
        for item in prefs[other]:
            
            # only score movies I haven't seen yet
            if item not in prefs[person] or prefs[person][item]==0:
                # Similarity * Score
                totals.setdefault(item,0)
                totals[item]+=prefs[other][item]*sim
                # Sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+=sim
  
    # Create the normalized list
    rankings=[(total/simSums[item],item) for item,total in totals.items()]
  
    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings

def ext_getRecommendationsSim(prefs, person, usersim, removed_item, weight, thresh):
    '''
        Calculates recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- usersim: previously calculated (within simu) user-user similarity matrix
        
        Returns:
        -- A rankings list with only the item removed by the LOOCV process,
            or an empty list if no recommendation calculated.
        
    '''
    totals={}
    simSums={}
    for other in prefs:
      # don't compare me to myself
        if other == person: 
            continue
        
        #finds the similarity weight from the user-user similarity matrix that has already been calculated
        # previously -- sim = similarity(prefs,person,other)
        sim = 0
        for tupl in usersim[person]:
            if tupl[1] == other:
                sim = tupl[0]
        
    
        # ignore scores of zero or lower
        if sim <= 0: continue
        # threshold
        if sim < thresh: continue 

        for item in prefs[other]:
            
            # only score movies I haven't seen yet
            if item not in prefs[person] or prefs[person][item]==0:
                # Similarity * Score
                totals.setdefault(item,0)
                totals[item]+=prefs[other][item]*sim
                # Sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+=sim
  
    # Create the normalized list
    rankings=[(total/simSums[item],item) for item,total in totals.items()]
    single = []
    for tupl in rankings:
        if tupl[1] == removed_item:
            single.append(tupl)
  
    return single

def ext_getRecommendedItems(prefs, user, itemMatch, removed_item, weight, thresh):
    '''
        Calculates recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A rankings list with only the item removed by the LOOCV process,
            or an empty list if no recommendation calculated.
        
    '''    

    userRatings=prefs[user]
    scores={}
    totalSim={}
    # Loop over items rated by this user
    for (item,rating) in userRatings.items( ):
  
      # Loop over items similar to this one
        for (similarity,item2) in itemMatch[item]:
    
            # Ignore if this user has already rated this item
            if item2 in userRatings: continue
            # ignore scores of zero or lower
            if similarity<=0: continue 
            # test threshold           
            if similarity<thresh: continue 
            # Weighted sum of rating times similarity
            scores.setdefault(item2,0)
            scores[item2]+=similarity*rating
            # Sum of all the similarities
            totalSim.setdefault(item2,0)
            totalSim[item2]+=similarity
  
    # Divide each total score by total weighting to get an average

    rankings=[(score/totalSim[item],item) for item,score in scores.items( )]    
  
    single = []
    for tupl in rankings:
        if tupl[1] == removed_item:
            single.append(tupl)
  
    return single
    
def get_all_UU_recs(prefs, sim=sim_pearson, num_users=10, top_N=5):
    ''' 
    Print user-based CF recommendations for all users in dataset

    Parameters
    -- prefs: nested dictionary containing a U-I matrix
    -- sim: similarity function to use (default = sim_pearson)
    -- num_users: max number of users to print (default = 10)
    -- top_N: max number of recommendations to print per user (default = 5)

    Returns: None
    '''
    i = 0
    for user in prefs.keys(): 
        recs = getRecommendations(prefs, user, sim)
        print ('User-based CF recs for %s, sim_pearson: ' % (user), 
                       recs[:top_N]) 
        i += 1
        if i >= num_users: 
            break

def loo_cv(prefs, metric, sim, algo):
    """
    Leave_One_Out Evaluation: evaluates recommender system ACCURACY
     
     Parameters:
         prefs dataset: critics, ml-100K, etc.
         metric: MSE, MAE, RMSE, etc.
         sim: distance, pearson, etc.
         algo: user-based recommender, item-based recommender, etc.
     
    Returns:
         error_total: MSE, MAE, RMSE totals for this set of conditions
         error_list: list of actual-predicted differences
    
    
    Algo Pseudocode ..
    Create a temp copy of prefs
    
    For each user in temp copy of prefs:
      for each item in each user's profile:
          delete this item
          get recommendation (aka prediction) list
          restore this item
          if there is a recommendation for this item in the list returned
              calc error, save into error list
          otherwise, continue
      
    return mean error, error list

    """
    error_list = []
    for user in prefs.keys():
        for item in prefs[user].keys():
            actual = prefs[user][item]
            temp = copy.deepcopy(prefs)
            temp[user].pop(item) # remove item 
            recs = algo(temp, user, sim) # get recommendation
            # prefs[user][item] = actual_rating # restore item]
            for rec in recs: 
                if rec[1] == item: 
                    prediction = rec[0]
                    error = (actual - prediction)**2
                    error_list.append(error)
                    print("User: %s, Item: %s, Prediction: %.3f, Actual: %.3f, Sq Error: %.3f " % (user, item, prediction, actual, error))
            
            if item not in set([rec[1] for rec in recs]):
                print("No prediction/recommendation available for User: %s, Item: %s " % (user, item))
                continue
    
    
    error = np.average(error_list)

    print("MSE for critics: %.3f using %s " % (error, str(sim)))
    
    return error, error_list

def calc_user_dist_similarities(prefs):
    user_similarities = {}
    for user1 in prefs.keys():
        for user2 in prefs.keys():
            if user1 != user2: 
                user_similarities[(user1, user2)] = sim_distance(prefs, user1, user2)
                print('Distance sim %s & %s: %.3f' % (user1, user2, user_similarities[(user1, user2)]))
    
    return user_similarities 

def calc_user_pc_similarities(prefs):

    user_similarities = {}
    for user1 in prefs.keys():
        for user2 in prefs.keys():
            if user1 != user2: 
                user_similarities[(user1, user2)] = sim_pearson(prefs, user1, user2)
                print('Pearson sim %s & %s: %.3f' % (user1, user2, user_similarities[(user1, user2)]))
    
    return user_similarities 

def topMatches(prefs,person,weight,counter,similarity=sim_pearson, n=5):
    '''
        Returns the best matches for person from the prefs dictionary

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        -- n: number of matches to find/return (5 is default)
        
        Returns:
        -- A list of similar matches with 0 or more tuples, 
           each tuple contains (similarity, item name).
           List is sorted, high to low, by similarity.
           An empty list is returned when no matches have been calc'd.
        
    '''     
    scores=[(similarity(prefs,person,other, weight),other) 
                    for other in prefs if other!=person]
    scores.sort()
    scores.reverse()
    # if counter < 10:
    #     print("itemsim: %s" % person)
    #     print(scores[0:n])

    return scores[0:n]

def transformPrefs(prefs):
    '''
        Transposes U-I matrix (prefs dictionary) 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        
        Returns:
        -- A transposed U-I matrix, i.e., if prefs was a U-I matrix, 
           this function returns an I-U matrix
        
    '''     
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})
            # Flip item and person
            result[item][person]=prefs[person][item]
    return result

def calculateSimilarItems(prefs, weight, n=100,similarity=sim_pearson):
    '''
        Creates a dictionary of items showing which other items they are most 
        similar to. 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- n: number of similar matches for topMatches() to return
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A dictionary with a similarity matrix
        
    '''
    result={}
    # Invert the preference matrix to be item-centric
    itemPrefs=transformPrefs(prefs)
    c=0
    for item in itemPrefs:
      # Status updates for larger datasets
        c+=1
        if c%100==0: 
            print("%s / %d" % (c, len(itemPrefs)))
            
        # Find the most similar items to this one
        scores=topMatches(itemPrefs,item, weight, c, similarity,n=n)
        result[item]=scores
        
    return result

def calculateSimilarUsers(prefs, weight, n=100,similarity=sim_pearson):

    '''
        Creates a dictionary of items showing which other items they are most 
        similar to. 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- n: number of similar matches for topMatches() to return
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A dictionary with a similarity matrix
        
    '''     
    result={}
    c=0
    for user in prefs:
      # Status updates for larger datasets
        c+=1
        if c%100==0: 
            print("%d / %d" % (c, len(prefs)))
            
        # Find the most similar items to this one
        scores=topMatches(prefs,user,weight, c, similarity,n=n)
        result[user]=scores
    return result

def getRecommendedItems(prefs, user, itemMatch, weight):
    '''
        Calculates recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''    

    userRatings=prefs[user]
    scores={}
    totalSim={}
    # Loop over items rated by this user
    for (item,rating) in userRatings.items( ):
  
      # Loop over items similar to this one
        for (similarity,item2) in itemMatch[item]:
    
            # Ignore if this user has already rated this item
            if item2 in userRatings: continue
            # ignore scores of zero or lower
            if similarity<=0: continue            
            # Weighted sum of rating times similarity
            scores.setdefault(item2,0)
            scores[item2]+=similarity*rating*weight
            # Sum of all the similarities
            totalSim.setdefault(item2,0)
            totalSim[item2]+=similarity*weight
  
    # Divide each total score by total weighting to get an average

    rankings=[(score/totalSim[item],item) for item,score in scores.items( )]    
  
    # Return the rankings from highest to lowest
    rankings.sort( )
    rankings.reverse( )
    return rankings
           
def get_all_II_recs(prefs, itemsim, sim_method, num_users=10, top_N=5):
    ''' 
    Print item-based CF recommendations for all users in dataset

    Parameters
    -- prefs: U-I matrix (nested dictionary)
    -- itemsim: item-item similarity matrix (nested dictionary)
    -- sim_method: name of similarity method used to calc sim matrix (string)
    -- num_users: max number of users to print (integer, default = 10)
    -- top_N: max number of recommendations to print per user (integer, default = 5)

    Returns: None
    
    '''
    i = 0
    for user in prefs.keys(): 
        recs = getRecommendedItems(prefs, itemsim, user)
        print ('Item-based CF recs for %s, %s: ' % (user, sim_method), 
                       recs[:top_N]) 
        i += 1
        if i >= num_users: 
            break

def loo_cv_sim(prefs, sim, algo, sim_matrix, weight, thresh, runs, k):
    """
    Leave-One_Out Evaluation: evaluates recommender system ACCURACY
     
     Parameters:
         prefs dataset: critics, etc.
     metric: MSE, or MAE, or RMSE
     sim: distance, pearson, etc.
     algo: user-based recommender, item-based recommender, etc.
     sim_matrix: pre-computed similarity matrix
     
    Returns:
         error: MSE, MAE, RMSE totals for this set of conditions
        mse_error_list: SE list for set of conditions

    """

    mse_error_list, mae_error_list = [], []
    i = 0
    start_time = timeit.default_timer()
    temp = copy.deepcopy(prefs)
    for user in prefs.keys():
        i += 1
        # if i%50 == 0:
        #     print("%d / %d" % (i, len(prefs.keys())))
        for item in list(prefs[user].keys()):
            #actual = prefs[user][item]
            actual = temp[user].pop(item) # remove item 
            recs = algo(temp, user, sim_matrix, item, weight, thresh) # get recommendation
            temp[user][item] = actual # restore item]
            for rec in recs: 
                
                if rec[1] == item: 
                    prediction = rec[0]
                    mse_error = (prediction - actual) **2 
                    mae_error = np.abs(prediction - actual)
                    mse_error_list.append(mse_error)
                    mae_error_list.append(mae_error)
                    
                    # print("User: %s, Item: %s, Prediction: %.3f, Actual: %.3f, Error: %.3f " % (user, item, prediction, actual, error))
            
            # if item not in set([rec[1] for rec in recs]):
            #         # print("From loo_cv_sim(), No prediction calculated for item %s, user %s in pred_list: %s" % (item, user, recs))
            #         continue
        if len(mse_error_list) > 0:
            if i%10 == 0:
                print("(%d / %d) Number of users processed: %d " % (k, runs, i))
                cur_time = timeit.default_timer()
                print("%.2f secs for %d users, time per user: %.2f" % (cur_time - start_time, i, (cur_time - start_time)/i))
                # print("Squared Error List:", mse_error_list)
                mse = sum(mse_error_list)/len(mse_error_list) 
                mae = sum(mae_error_list)/len(mae_error_list)
                rmse = math.sqrt(sum(mse_error_list)/len(mse_error_list))
                print("MSE: %f, RMSE: %f, MAE: %f" %(mse, rmse, mae))

    if len(mse_error_list) == 0:
        return 0, 0, 0, []
    mse = sum(mse_error_list)/len(mse_error_list)
    mae = sum(mae_error_list)/len(mae_error_list)
    rmse = math.sqrt(sum(mse_error_list)/len(mse_error_list))
    
    
    return mse, mae, rmse, mse_error_list

def main():
    ''' User interface for Python console '''

    # Load critics dict from file
    path = os.getcwd() # this gets the current working directory
                       # you can customize path for your own computer here
    print('\npath: %s' % path) # debug
    done = False
    prefs = {}
    
    while not done: 
        print()
        # Start a simple dialog
        file_io = input('R(ead) critics data from file?, \n'
                        'RML(ead ml100K data)?, \n'
                        'P(rint) the U-I matrix?, \n'
                        'V(alidate) the dictionary?, \n'
                        'S(tats) print?, \n'
                        'D(istance) critics data?, \n'
                        'PC(earson Correlation) critics data?, \n'
                        'U(ser-based CF Recommendations)?, \n'
                        'LCV(eave one out cross-validation)?, \n'
                        'SIM(ilarity matrix) calc for Item-based recommender?, \n'
                        'SIMU(ser-User matrix) calc for User-based recommender?, \n'
                        'I(tem-based CF Recommendations)?, \n'
                        'EXP(eriment)?, \n'
                        'LCVSIM(eave one out cross-validation)? \n ==> ')
        
        if file_io == 'R' or file_io == 'r':
            print()
            file_dir = 'data/'
            datafile = 'critics_ratings.data'
            itemfile = 'critics_movies.item'
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            print('Number of users: %d\nList of users:' % len(prefs), 
                  list(prefs.keys()))      

        elif file_io == 'RML' or file_io == 'rml':
            print()
            file_dir = 'data/ml-100k/' # path from current directory
            datafile = 'u.data'  # ratings file
            itemfile = 'u.item'  # movie titles file            
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            print('Number of users: %d\nList of users [0:10]:' 
                      % len(prefs), list(prefs.keys())[0:10] )  

        elif file_io == 'D' or file_io == 'd':
            print()
            if len(prefs) > 0:            
                # print('Examples:')
                # print ('Distance sim Lisa & Gene:', sim_distance(prefs, 'Lisa', 'Gene')) # 0.29429805508554946
                # num=1
                # den=(1+ sqrt( (2.5-3.0)**2 + (3.5-3.5)**2 + (3.0-1.5)**2 + (3.5-5.0)**2 + (3.0-3.0)**2 + (2.5-3.5)**2))
                # print('Distance sim Lisa & Gene (check):', num/den)    
                # print ('Distance sim Lisa & Michael:', sim_distance(prefs, 'Lisa', 'Michael')) # 0.4721359549995794
                # print()
                print('User-User distance similarities:')
                user_similarities = calc_user_dist_similarities(prefs)
                print()
                min_similarity = min(user_similarities, key=user_similarities.get)
                max_similarity = max(user_similarities, key=user_similarities.get)
                print("Least Similar Users are %s & %s, with RS distance of %.3f" % (min_similarity[0], min_similarity[1], 
                    user_similarities[min_similarity[0], min_similarity[1]]))
                print("Most Simiar Users: %s & %s, with RS distance of %.3f" % (max_similarity[0], max_similarity[1],
                    user_similarities[max_similarity[0], max_similarity[1]]))
                
            else:
                print ('Empty dictionary, R(ead) in some data!')    
        elif file_io == 'P' or file_io == 'p':
            # print the u-i matrix
            print()
            if len(prefs) > 0:
                print ('Printing "%s" dictionary from file' % datafile)
                print ('User-item matrix contents: user, item, rating')
                for user in prefs:
                    for item in prefs[user]:
                        print(user, item, prefs[user][item])
            else:
                print ('Empty dictionary, R(ead) in some data!')               
        elif file_io == 'V' or file_io == 'v':      
            print()
            if len(prefs) > 0:
                # Validate the dictionary contents ..
                print ('Validating "%s" dictionary from file' % datafile)
                print ("critics['Lisa']['Lady in the Water'] =", 
                       prefs['Lisa']['Lady in the Water']) # ==> 2.5
                print ("critics['Toby']:", prefs['Toby']) 
                # ==> {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 
                #      'Superman Returns': 4.0}
            else:
                print ('Empty dictionary, R(ead) in some data!')              
        elif file_io == 'S' or file_io == 's':
            print()
            filename = 'critics_ratings.data'
            if len(prefs) > 0:
                data_stats(prefs, filename)
                popular_items(prefs, filename)
            else: # Make sure there is data  to process ..
                print ('Empty dictionary, R(ead) in some data!')   
        elif file_io == 'PC' or file_io == 'pc':
            print()
            if len(prefs) > 0:             
                print ('Example:')
                print ('Pearson sim Lisa & Gene:', sim_pearson(prefs, 'Lisa', 'Gene')) # 0.39605901719066977
                print()
                
                print('Pearson for all users:')
                # Calc Pearson for all users
                
                print('User-User distance similarities:')
                user_similarities = calc_user_pc_similarities(prefs)
                print()
                min_similarity = min(user_similarities, key=user_similarities.get)
                max_similarity = max(user_similarities, key=user_similarities.get)

                print("Least Similar Users are %s & %s, with pearson correlation of %.3f" % (min_similarity[0], min_similarity[1], 
                    user_similarities[min_similarity[0], min_similarity[1]]))
                print("Most Simiar Users: %s & %s, with pearson correlation  of %.3f" % (max_similarity[0], max_similarity[1],
                    user_similarities[max_similarity[0], max_similarity[1]]))
                
                print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!')    
        elif file_io == 'U' or file_io == 'u':
            print()
            if len(prefs) > 0:             
                print ('Example:')
                user_name = 'Toby'
                print ('User-based CF recs for %s, sim_pearson: ' % (user_name), 
                       getRecommendations(prefs, user_name)) 
                        # [(3.3477895267131017, 'The Night Listener'), 
                        #  (2.8325499182641614, 'Lady in the Water'), 
                        #  (2.530980703765565, 'Just My Luck')]
                print ('User-based CF recs for %s, sim_distance: ' % (user_name),
                       getRecommendations(prefs, user_name, similarity=sim_distance)) 
                        # [(3.457128694491423, 'The Night Listener'), 
                        #  (2.778584003814924, 'Lady in the Water'), 
                        #  (2.422482042361917, 'Just My Luck')]
                print()
                
                print('User-based CF recommendations for all users:')
                # Calc User-based CF recommendations for all users
        
                ## add some code here to calc User-based CF recommendations 
                ## write a new function to do this ..
                print("Using sim_pearson:")
                get_all_UU_recs(prefs)
                print()
                get_all_UU_recs(prefs, sim_distance)
                
                ##    '''
                
                print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!')     
        # Testing the code ..
        elif file_io == 'LCV' or file_io == 'lcv':
            print()
            if len(prefs) > 0:             
                print ('LOO_CV Evaluation: ')            
                ## add some code here to calc LOOCV 
                ## write a new function to do this ..
                error, error_list = loo_cv(prefs, 'MSE', sim_pearson, getRecommendations)
                print()
                error, error_list = loo_cv(prefs, 'MSE', sim_distance, getRecommendations)
            else:
                print ('Empty dictionary, R(ead) in some data!')            
        
 
        elif file_io == 'I' or file_io == 'i':
            print()
            if len(prefs) > 0 and len(itemsim) > 0:                
                print ('Example:')
                user_name = 'Toby'
    
                print ('Item-based CF recs for %s, %s: ' % (user_name, sim_method), 
                       getRecommendedItems(prefs, itemsim, user_name)) 
                
                print()
                
                print('Item-based CF recommendations for all users:')
                
                get_all_II_recs(prefs, itemsim, sim_method) # num_users=10, and top_N=5 by default  '''
                
                print()
                
            else:
                if len(prefs) == 0:
                    print ('Empty dictionary, R(ead) in some data!')
                else:
                    print ('Empty similarity matrix, use Sim(ilarity) to create a sim matrix!')    
        
        elif file_io == 'Sim' or file_io == 'sim' or file_io == 'SIM':
            print()
            if len(prefs) > 0: 
                ready = False # sub command in progress
                sub_cmd = input('RD(ead) distance or RP(ead) pearson or WD(rite) distance or WP(rite) pearson? ')
                try:
                    if sub_cmd == 'RD' or sub_cmd == 'rd':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(open( "save_itemsim_distance.p", "rb" ))
                        sim_method = 'sim_distance'
                        recType = 'item'
    
                    elif sub_cmd == 'RP' or sub_cmd == 'rp':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(open( "save_itemsim_pearson.p", "rb" ))  
                        sim_method = 'sim_pearson'
                        recType = 'item'
                        
                    elif sub_cmd == 'WD' or sub_cmd == 'wd':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        weight = 50
                        itemsim = calculateSimilarItems(prefs, weight,similarity=sim_distance)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open( "save_itemsim_distance.p", "wb" ))
                        sim_method = 'sim_distance'
                        recType = 'item'
                        
                    elif sub_cmd == 'WP' or sub_cmd == 'wp':
                        weight = 50
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(prefs, weight, similarity=sim_pearson)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open( "save_itemsim_pearson.p", "wb" )) 
                        sim_method = 'sim_pearson'
                        recType = 'item'
                    
                    else:
                        print("Sim sub-command %s is invalid, try again" % sub_cmd)
                        continue
                    
                    ready = True # sub command completed successfully
                    
                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!'
                           ' Enter Sim(ilarity matrix) again and choose a Write command')
                    print()
                

                if len(itemsim) > 0 and ready == True: 
                    # Only want to print if sub command completed successfully
                    print ('Similarity matrix based on %s, len = %d' 
                           % (sim_method, len(itemsim)))
                    print()
                    # for item in itemsim:
                    #     print(item, itemsim[item])
                print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!') 
        elif file_io == 'Simu' or file_io == 'simu':
            print()
            if len(prefs) > 0: 
                ready = False # sub command in progress
                sub_cmd = input('RD(ead) distance or RP(ead) pearson or WD(rite) distance or WP(rite) pearson? ')
                try:
                    if sub_cmd == 'RD' or sub_cmd == 'rd':
                        # Load the dictionary back from the pickle file.
                        usersim = pickle.load(open( "save_usersim_distance.p", "rb" ))
                        sim_method = 'sim_distance'
                        recType = 'user'
    
                    elif sub_cmd == 'RP' or sub_cmd == 'rp':
                        # Load the dictionary back from the pickle file.
                        usersim = pickle.load(open( "save_usersim_pearson.p", "rb" ))  
                        sim_method = 'sim_pearson'
                        recType = 'user'
                        
                    elif sub_cmd == 'WD' or sub_cmd == 'wd':
                        # transpose the U-I matrix and calc user-user similarities matrix
                        usersim = calculateSimilarUsers(prefs,n=100,similarity=sim_distance)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(usersim, open( "save_usersim_distance.p", "wb" ))
                        sim_method = 'sim_distance'
                        recType = 'user'
                        
                    elif sub_cmd == 'WP' or sub_cmd == 'wp':
                        # transpose the U-I matrix and calc user-user similarities matrix
                        usersim = calculateSimilarUsers(prefs,n=100,similarity=sim_pearson)
                        # Dump/save dictionary to a pickle file
                        pickle.dump(usersim, open( "save_usersim_pearson.p", "wb" )) 
                        sim_method = 'sim_pearson'
                        recType = 'user'
                    
                    else:
                        print("Simu sub-command %s is invalid, try again" % sub_cmd)
                        continue
                    
                    ready = True # sub command completed successfully
                    
                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!'
                           ' Enter Simu(ser-User matrix) again and choose a Write command')
                    print()
                if len(usersim) > 0 and ready == True: 
                    # Only want to print if sub command completed successfully
                    print ('Similarity matrix based on %s, len = %d' 
                           % (sim_method, len(usersim)))
                    print()
                    # for user in usersim:
                    #     print(user, usersim[user])
                print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!') 
        elif file_io == 'LCVSIM' or file_io == 'lcvsim':
            print()
            if recType == 'user':
                recsim = usersim
            elif recType == 'item':
                recsim = itemsim
            n = 100
            weight = 50
            thresh = 0.5
            if len(prefs) > 0 and recsim !={}:             
                
                if len(prefs) == 7:
                    prefs_name = 'critics'
                else:
                    prefs_name = 'ML-100k'
                
                if recType == 'user':
                    algo = ext_getRecommendationsSim
                    rec = 'User'
                elif recType == 'item':
                    algo = ext_getRecommendedItems 
                    rec ='Item'
                else:
                    print('Invalid Recommendation Type')
                
                if sim_method == 'sim_pearson': 
                    print('%s-based LOO_CV_SIM Evaluation:' % rec)
                    sim = sim_pearson
                    
                    print("%s-based, SIM: %s, SIG_WTG_CUTOFF: %d, SIM_THRESH: %f, NUM_NEIGHBORS: %d" %  (rec, str(sim), weight, thresh, n))
                    mse, mae, rmse, error_list  = loo_cv_sim(prefs, sim, algo, recsim, weight, thresh, 1, 1)
                    
                    print('%s for %s: %.5f, len(SE list): %d using %s ' % ("MSE", prefs_name, mse, len(error_list), sim) )
                    print('%s for %s: %.5f, len(SE list): %d using %s ' % ("MAE", prefs_name, mae, len(error_list), sim) )
                    print('%s for %s: %.5f, len(SE list): %d using %s ' % ("RMSE", prefs_name, rmse, len(error_list), sim) )
                    print()
                elif sim_method == 'sim_distance':
                    print('%s-based LOO_CV_SIM Evaluation:' % rec)
                    sim = sim_distance
                    print("%s-based, SIM: %s, SIG_WTG_CUTOFF: %d, SIM_THRESH: %f, NUM_NEIGHBORS: %d" %  (rec, str(sim), weight, thresh, n))
                    mse, mae, rmse, error_list  = loo_cv_sim(prefs, sim, algo, recsim, weight, thresh, 1, 1)
                    
                    print('%s for %s: %.5f, len(SE list): %d using %s ' % ("MSE", prefs_name, mse, len(error_list), sim) )
                    print('%s for %s: %.5f, len(SE list): %d using %s ' % ("MAE", prefs_name, mae, len(error_list), sim) )
                    print('%s for %s: %.5f, len(SE list): %d using %s ' % ("RMSE", prefs_name, rmse, len(error_list), sim) )
                    print()
                else:
                    print('Run Sim(ilarity matrix) command to create/load Sim matrix!')
                if prefs_name == 'critics':
                    print(error_list)
            else:
                print ('Empty dictionary, run R(ead) OR Empty Sim Matrix, run Sim!')

        elif file_io == 'exp' or file_io == 'EXP':
            results = pd.DataFrame(columns=["Algorithm", "Sim. Method", "Sig. Weighting", "Sim. Threshold", "MSE", "MAE", "RMSE", "len(SE list)"])
            n = 100
            k = 0
            sim_methods = [sim_pearson, sim_distance, sim_cosine]
            rec_algorithms = [ext_getRecommendationsSim, ext_getRecommendedItems]
            sim_weightings = [1, 25, 50]
            sim_thresholds = [0, 0.3, 0.5]
            runs = len(sim_methods)*len(rec_algorithms)*len(sim_weightings)*len(sim_thresholds)
           
              
            for sim in sim_methods: 
                if sim == sim_distance:
                    sim_str = "distance"
                elif sim == sim_cosine:
                    sim_str = "cosine"
                elif sim == sim_jaccard: 
                    sim_str = "jaccard"
                elif sim == sim_kendall_tau: 
                    sim_str = "kendal tau"
                elif sim == sim_spearman: 
                    sim_str = 'spearman'
                elif sim == sim_tanimoto: 
                    sim_str = 'tanimoto'
                else: 
                    sim_str = "pearson"
                for algo in rec_algorithms: 
                    for weight in sim_weightings:
                        for thresh in sim_thresholds:
                            k+=1 
                            data = {}
                            
                            if algo == ext_getRecommendationsSim:
                                #sim_matrix = pickle.load(open( "save_usersim_distance.p", "rb" ))
                                sim_matrix = calculateSimilarUsers(prefs, weight,similarity=sim)
                                #pickle.dump(sim_matrix, open( "save_usersim_" + sim_str + ".p", "wb" ))
                                algo_str = 'User-based'
                                new_weight = False
                            else: 
                                sim_matrix = calculateSimilarItems(prefs, weight,similarity=sim)
                                #pickle.dump(sim_matrix, open( "save_itemsim_" + sim_str + ".p", "wb" ))
                                algo_str = 'Item-based'
                                new_weight = False
                            # else:
                            #     if algo == ext_getRecommendationsSim:
                            #         sim_matrix = pickle.load(open( "save_usersim_" + sim_str + ".p", "rb" ))
                            #         algo_str = 'User-based'
                            #     else:
                            #         sim_matrix = pickle.load(open( "save_itemsim_" + sim_str + ".p", "rb" ))
                            #         algo_str = 'Item-based'

                            print(algo_str, sim_str, weight, thresh)

                            mse, mae, rmse, error_list = loo_cv_sim(prefs, sim, algo, sim_matrix, weight, thresh, runs, len(results))
                            wtg_string = "n/" + str(weight)
                            data["Algorithm"], data['Sim. Method'], data['Sig. Weighting'], data['Sim. Threshold'] = algo_str, sim_str, wtg_string, thresh
                            data['MSE'], data['MAE'], data['RMSE'], data['len(SE list)'] = mse, mae, rmse, len(error_list)
                
                            results = results.append(data, ignore_index=True)
                            print("%d / %d" % (len(results), runs))

                            results.to_csv("experiment_results-" + str(k) + ".csv")  
        

        else:
            done = True
    
    print('\nGoodbye!')
        
if __name__ == '__main__':
    main()