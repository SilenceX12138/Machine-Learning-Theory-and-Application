tr_path = './data/covid.train.csv'  # path to training data
tt_path = './data/covid.test.csv'  # path to testing data

target_only = False  # TODO: Using 40 states & 2 tested_positive features

model_name = 'dnn' # nn/dnn

# TODO: How to tune these hyper-parameters to improve your model's performance?
config = {
    'n_epochs': 3000,  # maximum number of epochs
    'batch_size': 270,  # mini-batch size for dataloader
    'optimizer': 'SGD',  # optimization algorithm (optimizer in torch.optim)
    'optim_hparas':
    {  # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.001,  # learning rate of SGD
        'momentum': 0.9  # momentum for SGD
    },
    'early_stop':
    200,  # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'checkpoints/{}/{:.5}.pth'  # your model will be saved here
}
