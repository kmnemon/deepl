#import tushare as ts
import pymongo
from pymongo import MongoClient
import json
from datetime import datetime
import numpy as np
from predict import *


def connectTheclloction():
    client = MongoClient('mongodb://182.150.31.132:28998/')
    db = client.stockdb
    cl = db.stockdata

    return cl,db



def requireData(stock_id, all , start_time = '2015-01-01', end_time = '2018-03-21'):
    if( all == True):
        stock_data = ts.get_hist_data(str(stock_id))
    else:
        stock_data = ts.get_hist_data(str(stock_id), start=start_time, end=end_time)

    stock_data["date"] = ""
    stock_data["stock id"] = str(stock_id)
    data_time = stock_data.index

    data_time_translation = [datetime.strptime(d, '%Y-%m-%d').date() for d in data_time]
    n = data_time.size

    for i in range(n):
        stock_data.iat[i, 14] = str(data_time_translation[i])

    stock_data_json = stock_data.to_json(orient='records')

    #insert stockdata in db
    cl,db = connectTheclloction()

    cl.insert(json.loads(stock_data_json))
    cl.create_index([("date", pymongo.ASCENDING)], unique=True)

def stockRemove(stock_id):
    cl, db = connectTheclloction()
    cl.delete_many( {"stock id" : str(stock_id)})



def initialVecXY():
    cl, db = connectTheclloction()
    stock_data_doc = cl.find({"stock id" : "600848"})

    #m = train num     n = test num
    feature = 8
    n = 30
    m = stock_data_doc.count() - 1 - n

    # 14 = open, high, close...
    X_train_T = np.zeros(shape = (m, feature))
    X_test_T = np.zeros(shape = (30, feature))
    XTF = np.zeros(shape = (1, feature))
    train_y = np.zeros(shape = (1, m))
    test_y = np.zeros(shape = (1, n))

    #initial X
    i = 0
    j = 0
    flag = 0
    for doc in stock_data_doc:
        del doc['_id']
        del doc['date']
        del doc['stock id']
        del doc['price_change']
        del doc['p_change']
        del doc['volume']
        del doc['v_ma5']
        del doc['v_ma10']
        del doc['v_ma20']

        if( flag == 0):
            XTF = list(doc.values())
            flag = 1
            continue

    #assign X_test, n
        if( j < n):
            X_test_T[j] = list(doc.values())
            j = j + 1
    #assign X_train, m
        elif( i < m ):
            X_train_T[i] = list(doc.values())
            i = i + 1


    #initial Y
    i = 0
    j = 0
    flagy = 0
    flagx = 0
    for j in range(n):
        if( flagy == 0):
            x = lambda : XTF[2] > X_test_T[j][2]
            test_y[0][j] = x()
            flagy = 1
            continue
        x = lambda : X_test_T[j-1][2] > X_test_T[j][2]
        test_y[0][j] = x()
    for i in range(m):
        if (flagx == 0):
            x = lambda : X_test_T[n-1][2] > X_train_T[i][2]
            train_y[0][i] = x()
            flagx = 1
            continue
        x = lambda : X_train_T[i-1][2] > X_train_T[i][2]
        train_y[0][i] = x()


    train_x_orig = X_train_T.T
    test_x_orig = X_test_T.T


    return train_x_orig, train_y, test_x_orig, test_y

train_x_orig, train_y, test_x_orig, test_y = initialVecXY()
train_x = train_x_orig / 255
test_x = test_x_orig / 255

#print(train_x)

#print(test_x)




layers_dims = [8, 4, 1]

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations =25000, print_cost = True)
pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)


'''
### CONSTANTS ###
layers_dims = [12288, 20, 7, 5, 1] #  4-layer model

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations =2500, print_cost = True)
pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)
'''












#requireData(600848, all = True)
#stockRemove(600848)


#doc = cl.find_one({"word" : "oarlock"})
#doc["tt"] = "t1"

#cl.save(doc)


#cl.insert({"test2": "test3"})

#print(re)



#cl.insert(json.loads(a.to_json(orient='records')))



#r = cl.create_index([("word", pymongo.ASCENDING)], unique=True)
##a = sorted(list(db.wordcl.index_information()))
#print(a)

