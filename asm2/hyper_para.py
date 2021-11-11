class HyperPara:

    def __init__(self,
                 epoch =             None,
                 initializer =       None,
                 layer_tuple =       None,
                 activation =        None,
                 regularization_L1 = None,
                 regularization_L2 = None,
                 drop_out_rate =     None,
                 loss_func =         None,
                 optimizers =        None
                 ):

        self.epoch             =   epoch
        self.initializer       =   initializer
        self.layer_tuple       =   layer_tuple
        self.activation        =   activation
        self.regularization_L1 =   regularization_L1
        self.regularization_L2 =   regularization_L2
        self.drop_out_rate     =   drop_out_rate
        self.loss_func         =   loss_func
        self.optimizer        =   optimizers

