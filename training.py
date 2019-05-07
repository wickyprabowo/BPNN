import pandas as pd
from backpropagation import backpropagation

masukan = int(input("Amount of input layer : "))
hidden = int(input("Amount of hidden layer : "))
alpha = float(input("Amount of alpha : "))
max_epoch = int(input("Max of epoch : "))

df_train = pd.read_csv('datalatih.csv')
df_train = df_train.drop(df_train.columns[0], axis=1)
df_train = df_train.iloc[:, 0:masukan]
df_train_class = pd.read_csv('class_training.csv')
df_train_class = df_train_class.drop(df_train_class.columns[0], axis=1)
df_train_class.to_csv('class_training.csv')

bp = backpropagation(len(df_train.iloc[0]), hidden, len(df_train_class.iloc[0]), alpha, max_epoch)
bp.training(df_train.to_numpy(),df_train_class.to_numpy())

df_test = pd.read_csv('datatest.csv')
df_test = df_test.drop(df_test.columns[0], axis=1)
df_test = df_test.iloc[:, 0:masukan]
df_test_class = pd.read_csv('class_testing.csv')
df_test_class = df_test_class.drop(df_test_class.columns[0], axis=1)

bp.testing(df_test.to_numpy(), df_test_class.to_numpy())

# print(df_train.iloc[0])
# print(len(df_train.iloc[0]))
# print(df_train.iat[0,0])
# print(df_train.iloc[0][0])
# print(len(df_train.iloc[0]))
# print(df_train_class.iloc[0])
# print(df_test.to_numpy())