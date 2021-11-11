from asm2.hyper_para import HyperPara


hyper_para_dict = {
    1: HyperPara(epoch=2,
                 initializer="random_normal",
                 layer_tuple=(512, 512),
                 activation="relu",
                 regularization_L1=0,
                 regularization_L2=0,
                 drop_out_rate=0.2,
                 loss_func="categorical_crossentropy",
                 optimizers="sgd"),


}