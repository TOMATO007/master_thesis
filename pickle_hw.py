import os
os.getcwd()


import pickle
object_data = open('./data/hw_dataset.pickle', 'rb')
hw_data = pickle.load(object_data)


row_index = []
for i in range(len(hw_data)):
    n = 0
    for j in range(1, 25):
        if str(hw_data.iloc[i, j]) != 'nan':
            n = n + 1
    if n >= 24*0.5:
        row_index.append(i)
        
        
hw_data = hw_data.iloc[row_index,:]
hw_data


time_interval = ['2016-07-19 00:00:00', '2016-08-10 00:00:00', '2016-08-31 00:00:00', '2016-09-20 00:00:00', '2016-10-10 00:00:00', '2016-10-31 00:00:00']


valid_row_0 = (hw_data['time'] > time_interval[0]) & (hw_data['time'] < time_interval[1])
train_row_0 = valid_row_0 == False

valid_row_1 = (hw_data['time'] > time_interval[1]) & (hw_data['time'] < time_interval[2])
train_row_1 = valid_row_1 == False

valid_row_2 = (hw_data['time'] > time_interval[2]) & (hw_data['time'] < time_interval[3])
train_row_2 = valid_row_2 == False

valid_row_3 = (hw_data['time'] > time_interval[3]) & (hw_data['time'] < time_interval[4])
train_row_3 = valid_row_3 == False

valid_row_4 = (hw_data['time'] > time_interval[4]) & (hw_data['time'] < time_interval[5])
train_row_4 = valid_row_4 == False


valid_data_0 = hw_data[valid_row_0]
train_data_0 = hw_data[train_row_0]
valid_data_1 = hw_data[valid_row_1]
train_data_1 = hw_data[train_row_1]
valid_data_2 = hw_data[valid_row_2]
train_data_2 = hw_data[train_row_2]
valid_data_3 = hw_data[valid_row_3]
train_data_3 = hw_data[train_row_3]
valid_data_4 = hw_data[valid_row_4]
train_data_4 = hw_data[train_row_4]


print("the length of train_data_0 is %d" %len(train_data_0))
print("the length of valid_data_0 is %d" %len(valid_data_0))

print("the length of train_data_1 is %d" %len(train_data_1))
print("the length of valid_data_1 is %d" %len(valid_data_1))

print("the length of train_data_2 is %d" %len(train_data_2))
print("the length of valid_data_2 is %d" %len(valid_data_2))

print("the length of train_data_3 is %d" %len(train_data_3))
print("the length of valid_data_3 is %d" %len(valid_data_3))

print("the length of train_data_4 is %d" %len(train_data_4))
print("the length of valid_data_4 is %d" %len(valid_data_4))


import pickle

with open('./data/dataset_0/train_data_0.pickle', 'wb') as handle:
    pickle.dump(train_data_0, handle)
with open('./data/dataset_1/train_data_1.pickle', 'wb') as handle:
    pickle.dump(train_data_1, handle)
with open('./data/dataset_2/train_data_2.pickle', 'wb') as handle:
    pickle.dump(train_data_2, handle)
with open('./data/dataset_3/train_data_3.pickle', 'wb') as handle:
    pickle.dump(train_data_3, handle)
with open('./data/dataset_4/train_data_4.pickle', 'wb') as handle:
    pickle.dump(train_data_4, handle)

with open('./data/dataset_0/valid_data_0.pickle', 'wb') as handle:
    pickle.dump(valid_data_0, handle)
with open('./data/dataset_1/valid_data_1.pickle', 'wb') as handle:
    pickle.dump(valid_data_1, handle)
with open('./data/dataset_2/valid_data_2.pickle', 'wb') as handle:
    pickle.dump(valid_data_2, handle)
with open('./data/dataset_3/valid_data_3.pickle', 'wb') as handle:
    pickle.dump(valid_data_3, handle)
with open('./data/dataset_4/valid_data_4.pickle', 'wb') as handle:
    pickle.dump(valid_data_4, handle)
    