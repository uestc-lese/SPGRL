import configparser

class Config(object):
    def __init__(self, config_file):
        conf = configparser.ConfigParser()
        try:
            conf.read(config_file)
        except:
            print("loading config: %s failed" % (config_file))
        
        #Hyper-parameter
        self.epoch = conf.getint("Model_Setup", "epoch")
        self.lr = conf.getfloat("Model_Setup", "lr")
        self.weight_decay = conf.getfloat("Model_Setup", "weight_decay")
        self.k = conf.getint("Model_Setup", "k")
        self.nhid1 = conf.getint("Model_Setup", "nhid1")
        self.nhid2 = conf.getint("Model_Setup", "nhid2")
        self.dropout = conf.getfloat("Model_Setup", "dropout")
        self.feature_dimension = conf.getint("Model_Setup", "feature_dimension")       
        self.num_classes = conf.getint("Model_Setup", "num_classes")
        self.alpha = conf.getfloat("Model_Setup", "alpha")
        self.beta = conf.getfloat("Model_Setup", "beta")






        


