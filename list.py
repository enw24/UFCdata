#source: https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/?#
import csv
import pandas as pd, matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#trains model using difference of stats between two boxers to predict the winner
#output is the accuracy
def trainer(traindata, testdata):
    train_x = traindata.drop(columns=['Winner','B_current_lose_streak','B_current_win_streak','B_2','B_avg_SIG_STR_pct','B_losses','B_wins','B_Height_cms','B_Reach_cms','B_Weight_lbs','R_current_lose_streak','R_current_win_streak','R_2','R_avg_SIG_STR_pct','R_losses','R_wins','R_Height_cms','R_Reach_cms','R_Weight_lbs','B_age','R_age'],axis=1)
    train_y = traindata['Winner']

    test_x = testdata.drop(columns=['Winner','B_current_lose_streak','B_current_win_streak','B_2','B_avg_SIG_STR_pct','B_losses','B_wins','B_Height_cms','B_Reach_cms','B_Weight_lbs','R_current_lose_streak','R_current_win_streak','R_2','R_avg_SIG_STR_pct','R_losses','R_wins','R_Height_cms','R_Reach_cms','R_Weight_lbs','B_age','R_age'],axis=1)
    test_y = testdata['Winner']

    model = SVC()
    model.fit(train_x,train_y)
    predict_train = model.predict(train_x)
    #print('Target on train data',predict_train)

    accuracy_train = accuracy_score(train_y,predict_train)
    #print('accuracy_score on train dataset : ', accuracy_train)

    predict_test = model.predict(test_x)
    #print('Target on test data',predict_test)

    accuracy_test = accuracy_score(test_y,predict_test)
    print('accuracy_score on test dataset using differentail data: ', accuracy_test)
    #return predict_test


#trains model using the same number of stats about two boxers to predict the winner
#output is the accuracy
def trainnosubtraction(traindata, testdata):
    train_x = traindata.drop(columns=['Winner'],axis=1)
    train_y = traindata['Winner']

    test_x = testdata.drop(columns=['Winner'],axis=1)
    test_y = testdata['Winner']

    model = SVC()
    model.fit(train_x,train_y)
    predict_train = model.predict(train_x)
    #print('Target on train data',predict_train)

    accuracy_train = accuracy_score(train_y,predict_train)
    #print('accuracy_score on train dataset : ', accuracy_train)

    predict_test = model.predict(test_x)
    #print('Target on test data',predict_test)

    accuracy_test = accuracy_score(test_y,predict_test)
    print('accuracy_score on test dataset using all data: ', accuracy_test)
    #return predict_test


#runs program
def main():
    traindata = pd.read_csv('data2final.csv')
    testdata = pd.read_csv('datatest.csv')
    trainnosubtraction(traindata,testdata)

    traindata.insert(1,'windif',traindata["B_wins"].subtract(traindata["R_wins"]))
    traindata.insert(1, 'lossdif', traindata["B_losses"].subtract(traindata["R_losses"]))
    traindata.insert(1, 'winstreakif', traindata["B_current_win_streak"].subtract(traindata["R_current_win_streak"]))
    traindata.insert(1, 'losstreakdif', traindata["B_current_lose_streak"].subtract(traindata["R_current_lose_streak"]))
    traindata.insert(1, 'sigstrpctdif', traindata["B_avg_SIG_STR_pct"].subtract(traindata["R_avg_SIG_STR_pct"]))
    traindata.insert(1, 'tiedif', traindata["B_2"].subtract(traindata["R_2"]))
    traindata.insert(1, 'heightdif', traindata["B_Height_cms"].subtract(traindata["R_Height_cms"]))
    traindata.insert(1, 'weightdif', traindata["B_Weight_lbs"].subtract(traindata["R_Weight_lbs"]))
    traindata.insert(1, 'reachdif', traindata["B_Reach_cms"].subtract(traindata["R_Reach_cms"]))
    traindata.insert(1, 'agedif', traindata["B_age"].subtract(traindata["R_age"]))

    testdata.insert(1, 'windif', traindata["B_wins"].subtract(traindata["R_wins"]))
    testdata.insert(1, 'lossdif', traindata["B_losses"].subtract(traindata["R_losses"]))
    testdata.insert(1, 'winstreakif', traindata["B_current_win_streak"].subtract(traindata["R_current_win_streak"]))
    testdata.insert(1, 'losstreakdif', traindata["B_current_lose_streak"].subtract(traindata["R_current_lose_streak"]))
    testdata.insert(1, 'sigstrpctdif', traindata["B_avg_SIG_STR_pct"].subtract(traindata["R_avg_SIG_STR_pct"]))
    testdata.insert(1, 'tiedif', traindata["B_2"].subtract(traindata["R_2"]))
    testdata.insert(1, 'heightdif', traindata["B_Height_cms"].subtract(traindata["R_Height_cms"]))
    testdata.insert(1, 'weightdif', traindata["B_Weight_lbs"].subtract(traindata["R_Weight_lbs"]))
    testdata.insert(1, 'reachdif', traindata["B_Reach_cms"].subtract(traindata["R_Reach_cms"]))
    testdata.insert(1, 'agedif', traindata["B_age"].subtract(traindata["R_age"]))
    trainer(traindata, testdata)



if __name__=='__main__':
  main()
