labeled_path   = './data/training/labeled/'
unlabeled_path = './data/training/unlabeled/'
valid_path     = './data/validation/'
test_path      = './data/testing/'
model_path     = './checkpoints/models.pth'

SEED = 2000

batch_size    = 32
num_workers   = 2
n_epochs      = 80
learning_rate = 0.0001
momentum      = 0.8
weight_decay  = 1e-5

do_semi = False

model_name = 'resnet'

is_transfer = False
