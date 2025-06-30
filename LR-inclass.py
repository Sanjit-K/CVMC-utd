import matplotlib.pyplot as plt
import argparse
import csv
import random
import numpy as np
import copy
import pandas as pd


def compute_loss(pred_price, actual_price):
    loss = 0
    N = len(pred_price)
    for i in range(N):
        loss += (1/N) * (pred_price[i] - actual_price[i]) ** 2
    return loss

def compute_grad_a(train_sqfeet, pred_price, train_prices):
    grad_a = 0
    N = len(pred_price)
    for i in range(N):
        grad_a += (1/N) * (pred_price[i] - train_prices[i]) * train_sqfeet[i]
    grad_a *= 2
    return grad_a

def compute_grad_b(pred_price, train_prices):
    grad_b = 0
    N = len(pred_price)
    for i in range(N):
        grad_b += (1/N) * (pred_price[i] - train_prices[i])
    grad_b *= 2
    return grad_b



def parser():
    args_reader = argparse.ArgumentParser(description = 'scrip to train linear regresssion')
    args_reader.add_argument('--lr', type=float, default=0.01, help='learning rate')
    args_reader.add_argument('--csv_file_path', type=str, default='/Users/sanjitk./Desktop/CMCVutd/housing_train.csv', help='csv file path')
    args_reader.add_argument('--iterations', type=int, default=1000, help='number of iterations')
    args_reader.add_argument('--validation_interval', type=int, default = 10 ,help='validation interval')
    args = args_reader.parse_args()
    return args

def remove_outliers(allsqfeet, allprices):
    df = pd.DataFrame({'sqft': allsqfeet, 'price': allprices})

    Q1_sqft = df['sqft'].quantile(0.25)
    Q3_sqft = df['sqft'].quantile(0.75)
    IQR_sqft = Q3_sqft - Q1_sqft
    sqft_mask = (df['sqft'] >= Q1_sqft - 1.5 * IQR_sqft) & (df['sqft'] <= Q3_sqft + 1.5 * IQR_sqft)

    Q1_price = df['price'].quantile(0.25)
    Q3_price = df['price'].quantile(0.75)
    IQR_price = Q3_price - Q1_price
    price_mask = (df['price'] >= Q1_price - 1.5 * IQR_price) & (df['price'] <= Q3_price + 1.5 * IQR_price)

    df = df[sqft_mask & price_mask]

    allsqfeet = df['sqft'].tolist()
    allprices = df['price'].tolist()
    return allsqfeet, allprices


if __name__ == '__main__':
    args = parser()
    print ('training with learning rate', args.lr)
    print('Reading data from file: ', args.csv_file_path)
    file_handler = open(args.csv_file_path)
    csv_reader = csv.reader(file_handler, delimiter=',')
    all_lines = list(csv_reader)
    file_handler.close()

    all_lines = all_lines[1:]
    allsqfeet = [float(line[6]) for line in all_lines]
    allprices = [float(line[4]) for line in all_lines]
    print('Number of data points: ', len(allsqfeet))

    allsqfeet, allprices = remove_outliers(allsqfeet, allprices)

    #train, validation, test split
    total_N = len(allsqfeet)
    train_N = int(0.8 * total_N)
    val_N = int(0.1 * total_N)
    test_N = total_N - train_N + val_N
    test_sqfeet = allsqfeet[train_N + val_N:]
    train_prices = allprices[:train_N]
    train_sqfeet = allsqfeet[:train_N]
    val_prices = allprices[train_N:train_N + val_N]
    val_sqfeet = allsqfeet[train_N:train_N + val_N]
    test_prices = allprices[train_N + val_N:]

    #normalize
    
    max_sqft = max(train_sqfeet)
    max_price = max(train_prices)

    train_sqfeet = [(sqft) / (max_sqft) for sqft in train_sqfeet]
    train_prices = [(price) / (max_price) for price in train_prices]

    val_sqfeet = [(sqft) / (max_sqft) for sqft in val_sqfeet]
    val_prices = [(price) / (max_price) for price in val_prices]

    test_sqfeet = [(sqft) / (max_sqft) for sqft in test_sqfeet]
    test_prices = [(price) / (max_price) for price in test_prices]

    a_list = []
    b_list = []

    min_val_loss = int(1e9)
    
    #param initialization
    b = random.random()
    a = random.random()
    a_list.append(a)
    b_list.append(b)
    learnrate = args.lr
    iterations = args.iterations
    val_loss_list = []
    for t in range(iterations):
        pred_price = [a*sqft + b for sqft in train_sqfeet]
        loss = compute_loss(pred_price, train_prices)
        grad_a = compute_grad_a(train_sqfeet, pred_price, train_prices)
        grad_b = compute_grad_b(pred_price, train_prices)
        a = a - learnrate * grad_a
        b = b - learnrate * grad_b

        if t % args.validation_interval == 0:
            a_list.append(a)
            b_list.append(b)
            val_pred_price = [a*sqft + b for sqft in val_sqfeet]
            val_loss = compute_loss(val_pred_price, val_prices)
            val_loss_list.append(val_loss)
            min_val_loss = min(val_loss_list)
            print ('Epoch:', t, 'Minimum Validation Loss:', min_val_loss)


    
    pred_price = [a*sqft + b for sqft in test_sqfeet]
    loss = compute_loss(pred_price, test_prices)
    
    #plot scatter plot of val, and line according to a and b
    plt.subplot(1,2,1)
    plt.title('Actual vs Predicted')
    plt.xlabel('sqft')
    plt.ylabel('price')
    plt.plot(test_sqfeet, test_prices, 'ro') # actual values. 
    plt.plot(test_sqfeet, pred_price) #regressionl ine y = ax + b

    plt.subplot(1,2,2)
    #plot loss wrt epochs
    plt.xlabel('Epochs per 10')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.plot(val_loss_list, 'r')
    plt.show()





    

    
