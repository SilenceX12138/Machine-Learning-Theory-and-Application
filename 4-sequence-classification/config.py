path_config = {
    "data_dir"   : "./data/voxceleb",
    "save_path"  : "./checkpoints/{}/{:.5}.pth",
    "model_path" : "./checkpoints/{}/{}.pth",
    "output_path": "./predict.csv",
}

train_config = {
    "batch_size"  : 32,
    "n_workers"   : 0,
    "valid_steps" : 2000,
    "warmup_steps": 1000,
    "save_steps"  : 2000,
    "total_steps" : 70000,
    "model_name"  : 'conformer',
}

test_config = {
    'model_name': 'conformer',
}