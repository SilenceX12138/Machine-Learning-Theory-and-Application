from argparse import Namespace

from fairseq.tasks.translation import TranslationConfig, TranslationTask

config = Namespace(
    datadir     = "./data/dataset/data-bin/ted2020",
    savedir     = "./checkpoints/rnn",
    source_lang = "en",
    target_lang = "zh",

    # global seed
    seed = 1000,

    # cpu threads when fetching & processing data.
    num_workers = 0,
    # batch size in terms of tokens. gradient accumulation increases the effective batchsize.
    # max_tokens  = 8192,
    max_tokens = 4096,
    accum_steps = 2,

    # the lr s calculated from Noam lr scheduler. you can tune the maximum lr by this factor.
    lr_factor = 2.,
    lr_warmup = 4000,

    # clipping gradient norm helps alleviate gradient exploding
    clip_norm = 1.0,

    # maximum epochs for training
    max_epoch   = 30,
    start_epoch = 1,

    # beam size for beam search
    beam = 5,
    # generate sequences of maximum length ax + b, where x is the source length
    max_len_a = 1.2,
    max_len_b = 10,
    # when decoding, post process sentence by removing sentencepiece symbols and jieba tokenization.
    post_process = "sentencepiece",

    # checkpoints
    keep_last_epochs = 5,
    resume           = None, # if resume from checkpoint name (under config.savedir)

    # logging
    use_wandb = False,
)

## setup task
task_cfg = TranslationConfig(
    data                      = config.datadir,
    source_lang               = config.source_lang,
    target_lang               = config.target_lang,
    train_subset              = "train",
    required_seq_len_multiple = 8,
    dataset_impl              = "mmap",
    upsample_primary          = 1,
)
task = TranslationTask.setup_task(task_cfg)
task.load_dataset(split="train", epoch=1, combine=True) # combine if you have back-translation data.
task.load_dataset(split="valid", epoch=1)

arch_args = Namespace(
    encoder_embed_dim                = 256,
    encoder_ffn_embed_dim            = 512,
    encoder_layers                   = 1,
    decoder_embed_dim                = 256,
    decoder_ffn_embed_dim            = 1024,
    decoder_layers                   = 1,
    share_decoder_input_output_embed = True,
    dropout                          = 0.3,
)
