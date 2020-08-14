import warnings
import os


class DefaultConfig(object):
    base_dir = os.path.dirname(os.path.abspath(__file__)) # 项目文件夹

    env = os.path.join(base_dir, 'log_env') # tensorboard 环境

    # folder path
    train_data_folder = os.path.join(base_dir, './data/train') # 训练集存放路径
    checkpoints_folder = os.path.join(base_dir, './checkpoints') # checkpoint文件夹
    test_data_folder = os.path.join(base_dir, './data/test') # 测试集存放路径
    # load_model_path = 'checkpoints/model.pth' # 加载预训练的模型的路径，为None代表不加载

    # for data_loader
    batch_size = 4 # batch size
    num_workers = 4 # how many workers for loading data
    log_freq = 100 # print info every N batch
    print_freq = 1000
    pin_memory = True
    # correspond to pin_memory
    non_blocking = pin_memory
    
    debug_file = 'tmp/debug' # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'
        
    # for trianing
    use_gpu = True # use GPU or not
    num_epochs = 25
    # optimizer arguments
    lr = 0.001 # initial learning rate
    weight_decay = 0 
    # lr_decay = 0.95 # when val_loss increase, lr = lr*lr_decay

    def __init__(self):
        # 文件夹相关
        os.makedirs(self.checkpoints_folder, exist_ok=True)
        assert os.path.isdir(self.train_data_folder), \
               'train data folder \'{}\' is not a directory'.format(self.train_data_folder)

    def parse(self, kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                # warnings.warn('Warning: DefaultConfig does not have attribute {}'.format(key))
                print('Warning: DefaultConfig does not have attribute \'{}\' '.format(key))
            setattr(self, key, value)


if __name__ == '__main__':
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument('--test', type=int, default=0)
    parse.add_argument('--train', type=int,default=1)
    args = parse.parse_args()

    for k, v in vars(args).items():
        print(k, v)
    
    config = DefaultConfig()
    config.parse(vars(args))

    config.parse({'lr':0.003})
    print(config.__dict__)
    print(config.lr)