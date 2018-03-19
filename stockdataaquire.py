import tushare as ts
import pymongo

a = ts.get_hist_data('600848', start='2018-03-15',end='2018-03-16')
print(a.shape)