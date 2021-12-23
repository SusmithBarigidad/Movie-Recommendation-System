import pandas as pd
import numpy as np
from user import user
from scipy import spatial, stats
import math



def process_train(filename):
    data = pd.read_csv('train.txt', sep='\t', header=None)
    train_list = []
    for index, row in data.iterrows():
        train_list.append(user(index+1,data[row],data[row],0))
    return train_list
	
	
	
def statistics(userList):
    for userId in userList:
        userId.meanRating = userId.measure_meanRating()
        userId.sumSq = userId.measure_sumSq()
        userId.rmse = userId.measure_rmse()

    return userList
	


def statistics(userList):
    for userId in userList:
        userId.meanRating = userId.measure_meanRating()
        userId.sumSq = userId.measure_sumSq()
        userId.rmse = userId.measure_rmse()

    return userList
	

def process_test(filename):
    data_test = pd.read_csv(filename, sep=' ', header=None,dtype=str)
    data_test.columns = ['UserID','MovieID','Rating']
    test_list = []
    test_users = {}
    for index, row in data_test.iterrows():
        if data_test['UserID'][index] not in test_users:
            test_users[data_test['UserID'][index]]=user(data_test['UserID'][index],{data_test['MovieID'][index]:data_test['Rating'][index]},{data_test['MovieID'][index]:data_test['Rating'][index]},1)
        else:
            test_users[data_test['UserID'][index]].appendRating(data_test['MovieID'][index], int(data_test['Rating'][index]))


    for key in test_users:
        test_list.append(test_users[key])
    test_list = statistics(test_list)
    return test_list
	

def predictPearson(test_id, train_list, inv_ratings):
    test_user = test_id
    train_list = train_list
    processed_user = test_user

    for target in test_user.targets:
        if target in inv_ratings:
            n = 75
            top_n = {}
            total_similarity = 0
            min_key = str()
            for trainer_id in inv_ratings[target]:
                similarity = measurePearson(test_user, train_list[int(trainer_id)-1])
                if len(top_n) < n:
                    top_n[trainer_id] = similarity
                else:
                    top_n[trainer_id] = similarity
                    min_key = min(top_n, key=top_n.get)
                    top_n.pop(min_key)

            for trainer_id in top_n:
                total_similarity += int(top_n[trainer_id])
            big_term = 0
            for trainer_id in top_n:
                big_term += (int(top_n[trainer_id]) * ( train_list[int(trainer_id)-1].ratings[target] - train_list[int(trainer_id)-1].meanRating ))
            if total_similarity != 0:
                big_term = big_term/abs(total_similarity)
            else:
                big_term = 0
            processed_user.ratings[target] = int(round(test_user.meanRating + big_term))


    return processed_user
	

def measurePearson(test_id, train_id):
    test_list = []
    train_list = []
    numerator = 0
    for movie_id in test_id.ratings:
        if movie_id in train_id.ratings:
            if int(test_id.ratings[movie_id]) != 0 and int(train_id.ratings[movie_id]) != 0:
                test_list.append(int(test_id.ratings[movie_id]))
                train_list.append(int(train_id.ratings[movie_id]))

    if len(test_list) >= 2:
        array_test_list = np.asarray(test_list)
        array_train_list = np.asarray(test_list)
        result = pearsonCorr(array_test_list, array_train_list, test_id, train_id)
    else:
        result = 0

    return result
	


def pearsonCorr(x, y, test_id, train_id):
    x1 = x - test_id.meanRating
    y1 = y - train_id.meanRating
    denom = float(test_id.rmse * train_id.rmse)
    if denom != 0:
        return (x1 * y1).sum() / float(test_id.rmse * train_id.rmse)
    else:
        return 0
		
		
def resultOutput(result,path):
    with open(path,'w') as file:
        for tester_obj in result:
            for movie_id in tester_obj.ratings:
                if movie_id in tester_obj.targets:
                    output_ratings = tester_obj.ratings[movie_id]
                    if int(output_ratings) < 1:
                        output_ratings = 1
                    if int(output_ratings) > 5:
                        output_ratings = 5
                    file.write("{} {} {}\n".format(tester_obj.id, movie_id, output_ratings))
					






train_list = process_train('train.txt')

train_list = statistics(train_list)


inverted_list={}
for userID in train_list:
    for key in userID.ratings:
        if userID.ratings[key]!=0:
            if key in inverted_list:
                inverted_list[key].append(userID.id)
            else:
                inverted_list[key] = [userID.id]
				
				
for key in data_test.keys():
    
    result = []
    for testID in data_test[key]:
        result.append(predictPearson(testID, train_list, inverted_list))
    resultOutput(result,'pearson'+str(key))