from keras.models import Model
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D


# generic model design
def model_fn(actions):
    # unpack the actions from the list
    # 这里是每一层的超参数，目前是4层，如果想改变层数的话，就在这里写满足你层数的参数对
    # 比如你现在是想给一个8层的cnn，那么actions再加上kernel_i和filters_i，直到8，然后把下面的conv2D改掉，改成8层
    kernel_1, filters_1, kernel_2, filters_2, kernel_3, filters_3, kernel_4, filters_4 = actions
    # 将数据集图像输入，mnist数据集是（28，28，1），cifar10（32，32，3）根据需求做调整
    ip = Input(shape=(28, 28, 1))
    x = Conv2D(filters_1, (kernel_1, kernel_1), strides=(1, 1), padding='same', activation='relu')(ip)
    x = Conv2D(filters_2, (kernel_2, kernel_2), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters_3, (kernel_3, kernel_3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters_4, (kernel_4, kernel_4), strides=(1, 1), padding='same', activation='relu')(x)
    # x = Conv2D(filters_5, (kernel_5, kernel_5), strides=(1, 1), padding='same', activation='relu')(x)
    # x = Conv2D(filters_6, (kernel_6, kernel_6), strides=(1, 1), padding='same', activation='relu')(x)
    # x = Conv2D(filters_7, (kernel_7, kernel_7), strides=(1, 1), padding='same', activation='relu')(x)
    # x = Conv2D(filters_8, (kernel_8, kernel_8), strides=(1, 1), padding='same', activation='relu')(x)
    # x = Conv2D(filters_9, (kernel_9, kernel_9), strides=(1, 1), padding='same', activation='relu')(x)
    # x = Conv2D(filters_10, (kernel_10, kernel_10), strides=(1, 1), padding='same', activation='relu')(x)
    # 全局平均池化代替全连接层
    x = GlobalAveragePooling2D()(x)
    # 对数据集进行分类
    x = Dense(10, activation='softmax')(x)

    model = Model(ip, x)
    return model
