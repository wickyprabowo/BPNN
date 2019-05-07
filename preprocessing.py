import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('data.csv')

new_class = np.eye(13)
array_class_training = []
array_class_testing = []
for i in range(13):
    df_class = df.loc[:, 'class'] == (i+1)
    new_df_class_training = pd.DataFrame(new_class[i] for j in range(52))
    new_df_class_testing = pd.DataFrame([new_class[i] for j in range(23)])
    array_class_training.append(new_df_class_training)
    array_class_testing.append(new_df_class_testing)
concat_class_training = pd.concat(array_class_training)
concat_class_testing = pd.concat(array_class_testing)
concat_class_training.to_csv("class_training.csv")
concat_class_testing.to_csv("class_testing.csv")

selec_column = df.loc[:, 'roundness':'homogeneity']
df_class_1 = selec_column[df['class']==1]
train1, test1= train_test_split(df_class_1, test_size=0.3, random_state=4)
df_class_2= selec_column[df['class']==2]
train2, test2= train_test_split(df_class_2, test_size=0.3, random_state=4)
df_class_3= selec_column[df['class']==3]
train3, test3= train_test_split(df_class_3, test_size=0.3, random_state=4)
df_class_4= selec_column[df['class']==4]
train4, test4= train_test_split(df_class_4, test_size=0.3, random_state=4)
df_class_5= selec_column[df['class']==5]
train5, test5= train_test_split(df_class_5, test_size=0.3, random_state=4)
df_class_6= selec_column[df['class']==6]
train6, test6= train_test_split(df_class_6, test_size=0.3, random_state=4)
df_class_7= selec_column[df['class']==7]
train7, test7= train_test_split(df_class_7, test_size=0.3, random_state=4)
df_class_8= selec_column[df['class']==8]
train8, test8= train_test_split(df_class_8, test_size=0.3, random_state=4)
df_class_9= selec_column[df['class']==9]
train9, test9= train_test_split(df_class_9, test_size=0.3, random_state=4)
df_class_10= selec_column[df['class']==10]
train10, test10= train_test_split(df_class_10, test_size=0.3, random_state=4)
df_class_11= selec_column[df['class']==11]
train11, test11= train_test_split(df_class_11, test_size=0.3, random_state=4)
df_class_12= selec_column[df['class']==12]
train12, test12= train_test_split(df_class_12, test_size=0.3, random_state=4)
df_class_13= selec_column[df['class']==13]
train13, test13= train_test_split(df_class_13, test_size=0.3, random_state=4)


df_train = pd.concat([train1,train2,train3,train4,train5,train6,train7,train8,train9,train10,train11,train12,train13])
df_train.to_csv("datalatih.csv")
print(len(df_train))
df_test = pd.concat([test1,test2,test3,test4,test5,test6,test7,test8,test9,test10,test11,test12,test13])
df_test.to_csv("datatest.csv")

df_train = pd.read_csv('datalatih.csv')
df_train = df_train.drop(df_train.columns[0], axis=1)
df_test = pd.read_csv('datatest.csv')
df_test = df_test.drop(df_test.columns[0], axis=1)
df_train_class = pd.read_csv('class_training.csv')
df_train_class = df_train_class.drop(df_train_class.columns[0], axis=1)
df_test_class = pd.read_csv('class_testing.csv')
df_test_class = df_test_class.drop(df_test_class.columns[0], axis=1)

df_train.to_csv('datalatih.csv')
df_test.to_csv('datatest.csv')
df_train_class.to_csv('class_training.csv')
df_test_class.to_csv(('class_testing.csv'))





