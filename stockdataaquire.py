import tushare as ts
import pymongo
from pymongo import MongoClient
import json
from datetime import datetime





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
    client = MongoClient('mongodb://182.150.31.132:28998/')
    db = client.stockdb
    cl = db.stockdata

    cl.insert(json.loads(stock_data_json))
    cl.create_index([("date", pymongo.ASCENDING)], unique=True)

def stockRemove(stock_id):
    client = MongoClient('mongodb://182.150.31.132:28998/')
    db = client.stockdb
    cl = db.stockdata
    cl.delete_many( {"stock id" : str(stock_id)})

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

