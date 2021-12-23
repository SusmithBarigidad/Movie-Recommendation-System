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
	
	
def cosineSimilarity(test_user, train_user):
    test_list = []
    train_list = []
    for movieID in test_user.ratings:
        if movieID in train_user.ratings:
            if int(test_user.ratings[movieID]) != 0 and int(train_user.ratings[movieID]) != 0:
                test_list.append(int(test_user.ratings[movieID]))
                train_list.append(int(train_user.ratings[movieID]))

    if len(test_list) >= 2:
        result =  1 - spatial.distance.cosine(test_list, train_list)
    else:
        result = 0

    if math.isnan(result):
        return 0
    else:
        return result

def predictCosine(test_id, train_list, inv_ratings):
    test_user = test_id
    train_list = train_list
    processed_user = test_user
    for target in test_user.targets:
        # filtering top k ID
        if target in inv_ratings:
            n = 90
            top_n = {}
            total_similarity = 0
            predict_rating = 0
            for trainer_id in inv_ratings[target]:
                similarity = cosineSimilarity(test_user, train_list[int(trainer_id)-1])
                if len(top_n) < n:
                    top_n[trainer_id] = similarity
                else:
                    top_n[trainer_id] = similarity
                    min_key = min(top_n, key=top_n.get)
                    top_n.pop(min_key)
            # calculating the prediction based on cosine similarity        
            count = 0
            for trainer_id in top_n:
                total_similarity += top_n[trainer_id]
                count+=1
            for trainer_id in top_n:
                if total_similarity != 0:
                    predict_rating += (top_n[trainer_id]*train_list[int(trainer_id)-1].ratings[target])
                    

            if total_similarity != 0:
                predict_rating = predict_rating/total_similarity
            elif total_similarity == 0:
                predict_rating = test_user.meanRating

            processed_user.ratings[target] = int(round(predict_rating))

    return processed_user
	
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
filename = ['test5.txt','test10.txt','test20.txt']
data_test = {}
for file in filename:
    
    data_test[file] = process_test(file)
	
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
        result.append(predictCosine(testID, train_list, inverted_list))
    resultOutput(result,'cosine'+str(key))