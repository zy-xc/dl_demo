import warnings
import os


class DefaultConfig(object):
    env = 'default' # visdom 环境
    model = 'AlexNet' # 使用的模型，名字必须与models/__init__.py中的名字一致

    # folder path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_data_folder = os.path.join(base_dir, './data/train') # 训练集存放路径
    checkpoints_folder = os.path.join(base_dir, './checkpoints') # checkpoint文件夹
    test_data_root = os.path.join(base_dir, './data/test') # 测试集存放路径
    # load_model_path = 'checkpoints/model.pth' # 加载预训练的模型的路径，为None代表不加载

    # for data_loader
    batch_size = 4 # batch size
    num_workers = 4 # how many workers for loading data
    print_freq = 2000 # print info every N batch
    
    debug_file = '/tmp/debug' # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'
        
    # for trianing
    use_gpu = True # use GPU or not
    num_epochs = 2
    # optimizer arguments
    lr = 0.001 # initial learning rate
    # lr_decay = 0.95 # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0 # 损失函数

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