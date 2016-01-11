# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 09:30:02 2016

@author: c152803
"""

#%% init
import csv as csv
import numpy as np
import sklearn as sklearn
from sklearn.ensemble import RandomForestClassifier

#%% Load Data
csv_file_object = csv.reader(open('train.csv', 'rb')) 	# Load in the csv file
header = csv_file_object.next() 						# Skip the fist line as it is a header
data=[] 												# Create a variable to hold the data

for row in csv_file_object: 							# Skip through each row in the csv file,
    data.append(row[0:]) 								# adding each row to the data variable
data = np.array(data) 	

#%% Feature engineering 


def extract_title(names):
    t=["Capt.","Col.","Don.","Dona","Dr.","Jonkheer","Lady","Major","Master","Miss","Mlle","Mme","Mr.","Mrs.","Ms","Rev","Sir","the Countess"]
    titles=[]
    for i in range(0,len(names)):
        for j in range(0,len(t)):
            if t[j] in names[i]:
                titles.append(j)
                break
        
        else:
            titles.append("")
    return(titles)
    
def convert_sex(sexes):
    s=[]
    for i in range(0,len(sexes)):
        if("female" in sexes[i]):
            s.append("2")
        else:
           s.append("1") 
    return(s)
    
    
    #%%

data=np.c_[data,extract_title(data[0::,3])]
data=np.c_[data,convert_sex(data[0::,4])]
train=data[:,[0,2,6,7,9,12,13]]
train=train.astype(np.float)
target=data[0::,1]

#%% Train model


clf = RandomForestClassifier(max_depth=None, min_samples_split=1,random_state=0)
clf = clf.fit(train, target)


#%% Apply / Test Model
test_file = open('test.csv', 'rb')
test_file_object = csv.reader(test_file)
header_test = test_file_object.next()

test_data=[]
for row in test_file_object: 							# Skip through each row in the csv file,
    test_data.append(row[0:]) 								# adding each row to the data variable
test_data = np.array(test_data) 	



#%%
test_data=np.c_[test_data,extract_title(test_data[0::,1])]
test_data=np.c_[test_data,convert_sex(test_data[0::,2])]

dat_apply=test_data[:,[0,1,5,6,8,11,12]]
dat_apply=dat_apply.astype(np.float)

#%%
dat_apply=np.c_[dat_apply,clf.predict(dat_apply)]

#%%
predictions_file = open("first_model.csv", "wb")
predictions_file_object = csv.writer(predictions_file)
predictions_file_object.writerow(["PassengerId", "Survived"])	# write the column headers
for row in dat_apply:									# For each row in test file									
        predictions_file_object.writerow([int(row[0]), row[7]])			# write the PassengerId, and prediction
test_file.close()												# Close out the files.
predictions_file.close()
