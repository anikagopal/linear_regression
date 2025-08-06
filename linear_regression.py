import argparse, csv, random, copy
from matplotlib import pyplot as plt 

#define function
def parser(): 
    args_reader = argparse.ArgumentParser(description = 'Script to train a linear regression model with two parameters on house rent dataset.')
    args_reader.add_argument('--csv_file_path', type = str, default = '/Users/anikagopal/linear_regression/housing_train.csv', help = 'Path to the csv file containing house rent data.')
    args_reader.add_argument('--lr', type = float, default = 0.001, help = 'Learning rate for training the linear regression.')
    args_reader.add_argument('--epochs', type = int, default = 1000, help = 'Number of epochs to train the model on.')
    args_reader.add_argument('--validation_interval', type = int, default = 20, help = 'Number of updates between each validation.')
    args_reader.add_argument('--lr_drop_epoch', type = int, default = 800, help = 'Epoch at which lr drops.')
    args_reader.add_argument('--lr_drop_factor', type = float, default = 0.1, help = 'Factor by which the learning rate drops.')

    args = args_reader.parse_args()
    return args

#define function 
def compute_loss(pred, gt):
    total_loss = 0 
    N = len(pred)
    for ind in range(N):
        total_loss += (1/N) * (pred[ind] - gt[ind] ** 2)
    return total_loss

#define function
def compute_grad_a(train_sqfeet, pred_price, train_prices):
    grad_a = 0
    N = len(train_sqfeet) 
    for ind in range(N):
        grad_a += (1/N) * (pred_price[ind] - train_prices[ind]) * 2 * train_prices[ind]

    return grad_a

#define function
def compute_grad_b(pred_price, train_prices):
    grad_b = 0
    N = len(train_sqfeet) 
    for ind in range(N):
        grad_b += (1/N) * (pred_price[ind] - train_prices[ind]) * 2 

    return grad_b


if __name__ == "__main__":
    args = parser()  
    print('Training with learn rate: ', args.lr)
    print('Reading data from file: ', args.csv_file_path)
    file_handler = open(args.csv_file_path, 'r')
    csv_reader = csv.reader(file_handler)
    all_lines = list(csv_reader)
    file_handler.close() 

    all_lines = all_lines[1:]
    all_sqfeet = [float(x[6]) for x in all_lines]
    all_prices = [float(x[4]) for x in all_lines]
    
    #for ind in range (0,5): 
    #   print(all_sqfeet[ind], ' ', all_prices[ind])

    #Split data into train, validation, and test splits 
    total_N = len(all_sqfeet)
    train_N = int(0.8 * total_N)
    val_N = int(0.1 * total_N)
    test_N = total_N - (train_N +val_N)

    train_sqfeet = all_sqfeet[0:train_N]
    train_prices = all_sqfeet[0:train_N]

    max_sqfeet = max(train_sqfeet)
    max_price = max(train_prices)

    train_sqfeet = [elem/max_sqfeet for elem in train_sqfeet]
    train_prices = [elem/max_price for elem in train_prices]

    val_sqfeet = all_sqfeet[train_N : train_N + val_N]
    val_prices = all_prices[train_N : train_N + val_N]

    val_sqfeet = [elem/max_sqfeet for elem in val_sqfeet]
    val_prices = [elem/max_price for elem in val_prices] 
    
    test_sqfeet = all_sqfeet[train_N + val_N : ]
    test_prices = all_sqfeet[train_N + val_N : ]

    test_sqfeet = [elem/max_sqfeet for elem in test_sqfeet]
    test_prices = [elem/max_price for elem in test_prices] 


    a = random.random()
    b = random.random()

    a_lst = []
    b_lst = []
    val_loss_lst = [] 
    train_loss_lst = []

    min_val_loss = 10000
    min_val_loss_index = -1 

    print(f'Initial values a: {a}, b: {b}')

    for epoch in range(args.epochs):
        if epoch % args.validation_interval == 0: 
            a_lst.append(copy.deepcopy(a))
            b_lst.append(copy.deepcopy(b))
            val_pred_price = [a * val_sq_ft + b for val_sq_ft in val_sqfeet]
            val_loss = compute_loss(val_pred_price, val_prices)
            val_loss_lst.append(copy.deepcopy(val_loss))
            print(" ")
            print("val_loss: ", val_loss)
            print(" ")
            if val_loss < min_val_loss: 
                min_val_loss = val_loss
                min_val_loss_index = int(epoch/args.validation_interval)

        learning_rate_updated = False 
        if epoch > args.lr_drop_epoch and not learning_rate_updated: 
            args.lr = args.lr * args.lr_drop_factor
            learning_rate_updated = True 

        pred_price = [a * sqft + b for sqft in train_sqfeet] # Forward Pass
        loss = compute_loss(pred_price, train_prices)
        train_loss_lst.append(copy.deepcopy(loss))
        print('loss: ', loss)
        
        grad_a = compute_grad_a(train_sqfeet, pred_price, train_prices)
        grad_b = compute_grad_b(pred_price, train_prices)

        print('grad_a: ', grad_a)
        print('grad_b: ', grad_b)

        a = a - args.lr * grad_a 
        b = b - args.lr * grad_b 
        print('a: ', a)
        print('b: ', b)

    a_test = a_lst[min_val_loss_index]
    b_test = b_lst[min_val_loss_index]

    pred_price_test = [a_test * x + b_test for x in test_sqfeet]
    test_loss = compute_loss(pred_price_test, test_sqfeet) 
    print('final loss: ', test_loss)

    plt.plot(range(len(train_loss_lst)), train_loss_lst, label = 'train_loss', color = 'green')
    plt.plot(range(0, len(val_loss_lst) * args.validation_interval, args.validation_interval), val_loss_lst, label = 'val_loss', color = 'blue') 
              
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('train_val_loss.png')