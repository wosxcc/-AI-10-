import paddle
# import math
# from x2paddle.op_mapper.pytorch2paddle import pytorch_custom_layer as x2paddle_nn

class Preact_layer1__1_0_se_fc(paddle.nn.Layer):
    def __init__(self, linear0_in_features, linear0_out_features, linear1_in_features, linear1_out_features):
        super(Preact_layer1__1_0_se_fc, self).__init__()
        self.linear0 = paddle.nn.Linear(in_features=linear0_in_features, out_features=linear0_out_features)
        self.relu0 = paddle.nn.ReLU()
        self.linear1 = paddle.nn.Linear(in_features=linear1_in_features, out_features=linear1_out_features)
        self.x4 = paddle.nn.Sigmoid()
    def forward(self, x0):
        x1 = self.linear0(x0)
        x2 = self.relu0(x1)
        x3 = self.linear1(x2)
        x5 = self.x4(x3)
        return x5

class SELayer(paddle.nn.Layer):
    def __init__(self, x3_shape, preact_layer1__1_0_se_fc0_linear0_in_features, preact_layer1__1_0_se_fc0_linear0_out_features, preact_layer1__1_0_se_fc0_linear1_in_features, preact_layer1__1_0_se_fc0_linear1_out_features, x5_shape):
        super(SELayer, self).__init__()
        self.x1 = paddle.nn.AdaptiveAvgPool2D(output_size=[1, 1])
        self.x3_shape = x3_shape
        self.preact_layer1__1_0_se_fc0 = Preact_layer1__1_0_se_fc(linear0_in_features=preact_layer1__1_0_se_fc0_linear0_in_features, linear0_out_features=preact_layer1__1_0_se_fc0_linear0_out_features, linear1_in_features=preact_layer1__1_0_se_fc0_linear1_in_features, linear1_out_features=preact_layer1__1_0_se_fc0_linear1_out_features)
        self.x5_shape = x5_shape
    def forward(self, x0):
        x2 = self.x1(x0)
        b,c,_,_ = x2.shape
        x3 = paddle.reshape(x=x2, shape=(b,c))
        x4 = self.preact_layer1__1_0_se_fc0(x3)
        x5 = paddle.reshape(x=x4, shape=(b,c,1,1))
        x6 = x0 * x5
        return x6

class Preact_layer1__1_0_downsample(paddle.nn.Layer):
    def __init__(self, conv0_out_channels, conv0_stride, conv0_in_channels, bn0_num_channels):
        super(Preact_layer1__1_0_downsample, self).__init__()
        self.conv0 = paddle.nn.Conv2D(kernel_size=(1, 1), bias_attr=False, out_channels=conv0_out_channels, stride=conv0_stride, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, momentum=0.1, num_channels=bn0_num_channels)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2

class Bottleneck0(paddle.nn.Layer):
    def __init__(self, conv0_out_channels, conv0_in_channels, bn0_num_channels, conv1_out_channels, conv1_stride, conv1_in_channels, bn1_num_channels, conv2_out_channels, conv2_in_channels, bn2_num_channels, selayer0_x3_shape, selayer0_preact_layer1__1_0_se_fc0_linear0_in_features, selayer0_preact_layer1__1_0_se_fc0_linear0_out_features, selayer0_preact_layer1__1_0_se_fc0_linear1_in_features, selayer0_preact_layer1__1_0_se_fc0_linear1_out_features, selayer0_x5_shape, preact_layer1__1_0_downsample0_conv0_out_channels, preact_layer1__1_0_downsample0_conv0_stride, preact_layer1__1_0_downsample0_conv0_in_channels, preact_layer1__1_0_downsample0_bn0_num_channels):
        super(Bottleneck0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(kernel_size=(1, 1), bias_attr=False, out_channels=conv0_out_channels, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, momentum=0.1, num_channels=bn0_num_channels)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(kernel_size=(3, 3), bias_attr=False, padding=1, out_channels=conv1_out_channels, stride=conv1_stride, in_channels=conv1_in_channels)
        self.bn1 = paddle.nn.BatchNorm(is_test=True, momentum=0.1, num_channels=bn1_num_channels)
        self.relu1 = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(kernel_size=(1, 1), bias_attr=False, out_channels=conv2_out_channels, in_channels=conv2_in_channels)
        self.bn2 = paddle.nn.BatchNorm(is_test=True, momentum=0.1, num_channels=bn2_num_channels)
        self.selayer0 = SELayer(x3_shape=selayer0_x3_shape, preact_layer1__1_0_se_fc0_linear0_in_features=selayer0_preact_layer1__1_0_se_fc0_linear0_in_features, preact_layer1__1_0_se_fc0_linear0_out_features=selayer0_preact_layer1__1_0_se_fc0_linear0_out_features, preact_layer1__1_0_se_fc0_linear1_in_features=selayer0_preact_layer1__1_0_se_fc0_linear1_in_features, preact_layer1__1_0_se_fc0_linear1_out_features=selayer0_preact_layer1__1_0_se_fc0_linear1_out_features, x5_shape=selayer0_x5_shape)
        self.preact_layer1__1_0_downsample0 = Preact_layer1__1_0_downsample(conv0_out_channels=preact_layer1__1_0_downsample0_conv0_out_channels, conv0_stride=preact_layer1__1_0_downsample0_conv0_stride, conv0_in_channels=preact_layer1__1_0_downsample0_conv0_in_channels, bn0_num_channels=preact_layer1__1_0_downsample0_bn0_num_channels)
        self.relu2 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = self.selayer0(x8)
        x10 = self.preact_layer1__1_0_downsample0(x0)
        x11 = x9 + 1 * x10
        x12 = self.relu2(x11)
        return x12

class Bottleneck1(paddle.nn.Layer):
    def __init__(self, conv0_out_channels, conv0_in_channels, bn0_num_channels, conv1_out_channels, conv1_in_channels, bn1_num_channels, conv2_out_channels, conv2_in_channels, bn2_num_channels):
        super(Bottleneck1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(kernel_size=(1, 1), bias_attr=False, out_channels=conv0_out_channels, in_channels=conv0_in_channels)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, momentum=0.1, num_channels=bn0_num_channels)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(kernel_size=(3, 3), bias_attr=False, padding=1, out_channels=conv1_out_channels, in_channels=conv1_in_channels)
        self.bn1 = paddle.nn.BatchNorm(is_test=True, momentum=0.1, num_channels=bn1_num_channels)
        self.relu1 = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(kernel_size=(1, 1), bias_attr=False, out_channels=conv2_out_channels, in_channels=conv2_in_channels)
        self.bn2 = paddle.nn.BatchNorm(is_test=True, momentum=0.1, num_channels=bn2_num_channels)
        self.relu2 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = x8 + 1 * x0
        x10 = self.relu2(x9)
        return x10

class Preact_layer3(paddle.nn.Layer):
    def __init__(self, ):
        super(Preact_layer3, self).__init__()
        self.bottleneck00 = Bottleneck0(conv0_out_channels=256, conv0_in_channels=512, bn0_num_channels=256, conv1_out_channels=256, conv1_stride=[2, 2], conv1_in_channels=256, bn1_num_channels=256, conv2_out_channels=1024, conv2_in_channels=256, bn2_num_channels=1024, selayer0_x3_shape=[1, 1024], selayer0_preact_layer1__1_0_se_fc0_linear0_in_features=1024, selayer0_preact_layer1__1_0_se_fc0_linear0_out_features=1024, selayer0_preact_layer1__1_0_se_fc0_linear1_in_features=1024, selayer0_preact_layer1__1_0_se_fc0_linear1_out_features=1024, selayer0_x5_shape=[1, 1024, 1, 1], preact_layer1__1_0_downsample0_conv0_out_channels=1024, preact_layer1__1_0_downsample0_conv0_stride=[2, 2], preact_layer1__1_0_downsample0_conv0_in_channels=512, preact_layer1__1_0_downsample0_bn0_num_channels=1024)
        self.bottleneck10 = Bottleneck1(conv0_out_channels=256, conv0_in_channels=1024, bn0_num_channels=256, conv1_out_channels=256, conv1_in_channels=256, bn1_num_channels=256, conv2_out_channels=1024, conv2_in_channels=256, bn2_num_channels=1024)
        self.bottleneck11 = Bottleneck1(conv0_out_channels=256, conv0_in_channels=1024, bn0_num_channels=256, conv1_out_channels=256, conv1_in_channels=256, bn1_num_channels=256, conv2_out_channels=1024, conv2_in_channels=256, bn2_num_channels=1024)
        self.bottleneck12 = Bottleneck1(conv0_out_channels=256, conv0_in_channels=1024, bn0_num_channels=256, conv1_out_channels=256, conv1_in_channels=256, bn1_num_channels=256, conv2_out_channels=1024, conv2_in_channels=256, bn2_num_channels=1024)
        self.bottleneck13 = Bottleneck1(conv0_out_channels=256, conv0_in_channels=1024, bn0_num_channels=256, conv1_out_channels=256, conv1_in_channels=256, bn1_num_channels=256, conv2_out_channels=1024, conv2_in_channels=256, bn2_num_channels=1024)
        self.bottleneck14 = Bottleneck1(conv0_out_channels=256, conv0_in_channels=1024, bn0_num_channels=256, conv1_out_channels=256, conv1_in_channels=256, bn1_num_channels=256, conv2_out_channels=1024, conv2_in_channels=256, bn2_num_channels=1024)
    def forward(self, x0):
        x1 = self.bottleneck00(x0)
        x2 = self.bottleneck10(x1)
        x3 = self.bottleneck11(x2)
        x4 = self.bottleneck12(x3)
        x5 = self.bottleneck13(x4)
        x6 = self.bottleneck14(x5)
        return x6

class Preact_layer2(paddle.nn.Layer):
    def __init__(self, ):
        super(Preact_layer2, self).__init__()
        self.bottleneck00 = Bottleneck0(conv0_out_channels=128, conv0_in_channels=256, bn0_num_channels=128, conv1_out_channels=128, conv1_stride=[2, 2], conv1_in_channels=128, bn1_num_channels=128, conv2_out_channels=512, conv2_in_channels=128, bn2_num_channels=512, selayer0_x3_shape=[1, 512], selayer0_preact_layer1__1_0_se_fc0_linear0_in_features=512, selayer0_preact_layer1__1_0_se_fc0_linear0_out_features=512, selayer0_preact_layer1__1_0_se_fc0_linear1_in_features=512, selayer0_preact_layer1__1_0_se_fc0_linear1_out_features=512, selayer0_x5_shape=[1, 512, 1, 1], preact_layer1__1_0_downsample0_conv0_out_channels=512, preact_layer1__1_0_downsample0_conv0_stride=[2, 2], preact_layer1__1_0_downsample0_conv0_in_channels=256, preact_layer1__1_0_downsample0_bn0_num_channels=512)
        self.bottleneck10 = Bottleneck1(conv0_out_channels=128, conv0_in_channels=512, bn0_num_channels=128, conv1_out_channels=128, conv1_in_channels=128, bn1_num_channels=128, conv2_out_channels=512, conv2_in_channels=128, bn2_num_channels=512)
        self.bottleneck11 = Bottleneck1(conv0_out_channels=128, conv0_in_channels=512, bn0_num_channels=128, conv1_out_channels=128, conv1_in_channels=128, bn1_num_channels=128, conv2_out_channels=512, conv2_in_channels=128, bn2_num_channels=512)
        self.bottleneck12 = Bottleneck1(conv0_out_channels=128, conv0_in_channels=512, bn0_num_channels=128, conv1_out_channels=128, conv1_in_channels=128, bn1_num_channels=128, conv2_out_channels=512, conv2_in_channels=128, bn2_num_channels=512)
    def forward(self, x0):
        x1 = self.bottleneck00(x0)
        x2 = self.bottleneck10(x1)
        x3 = self.bottleneck11(x2)
        x4 = self.bottleneck12(x3)
        return x4

class Preact_layer1__1(paddle.nn.Layer):
    def __init__(self, bottleneck00_conv0_out_channels, bottleneck00_conv0_in_channels, bottleneck00_bn0_num_channels, bottleneck00_conv1_out_channels, bottleneck00_conv1_in_channels, bottleneck00_bn1_num_channels, bottleneck00_conv2_out_channels, bottleneck00_conv2_in_channels, bottleneck00_bn2_num_channels, bottleneck00_selayer0_x3_shape, bottleneck00_selayer0_preact_layer1__1_0_se_fc0_linear0_in_features, bottleneck00_selayer0_preact_layer1__1_0_se_fc0_linear0_out_features, bottleneck00_selayer0_preact_layer1__1_0_se_fc0_linear1_in_features, bottleneck00_selayer0_preact_layer1__1_0_se_fc0_linear1_out_features, bottleneck00_selayer0_x5_shape, bottleneck00_preact_layer1__1_0_downsample0_conv0_out_channels, bottleneck00_preact_layer1__1_0_downsample0_conv0_in_channels, bottleneck00_preact_layer1__1_0_downsample0_bn0_num_channels, bottleneck10_conv0_out_channels, bottleneck10_conv0_in_channels, bottleneck10_bn0_num_channels, bottleneck10_conv1_out_channels, bottleneck10_conv1_in_channels, bottleneck10_bn1_num_channels, bottleneck10_conv2_out_channels, bottleneck10_conv2_in_channels, bottleneck10_bn2_num_channels, bottleneck11_conv0_out_channels, bottleneck11_conv0_in_channels, bottleneck11_bn0_num_channels, bottleneck11_conv1_out_channels, bottleneck11_conv1_in_channels, bottleneck11_bn1_num_channels, bottleneck11_conv2_out_channels, bottleneck11_conv2_in_channels, bottleneck11_bn2_num_channels):
        super(Preact_layer1__1, self).__init__()
        self.bottleneck00 = Bottleneck0(conv0_out_channels=bottleneck00_conv0_out_channels, conv0_in_channels=bottleneck00_conv0_in_channels, bn0_num_channels=bottleneck00_bn0_num_channels, conv1_out_channels=bottleneck00_conv1_out_channels, conv1_stride=[1, 1], conv1_in_channels=bottleneck00_conv1_in_channels, bn1_num_channels=bottleneck00_bn1_num_channels, conv2_out_channels=bottleneck00_conv2_out_channels, conv2_in_channels=bottleneck00_conv2_in_channels, bn2_num_channels=bottleneck00_bn2_num_channels, selayer0_x3_shape=bottleneck00_selayer0_x3_shape, selayer0_preact_layer1__1_0_se_fc0_linear0_in_features=bottleneck00_selayer0_preact_layer1__1_0_se_fc0_linear0_in_features, selayer0_preact_layer1__1_0_se_fc0_linear0_out_features=bottleneck00_selayer0_preact_layer1__1_0_se_fc0_linear0_out_features, selayer0_preact_layer1__1_0_se_fc0_linear1_in_features=bottleneck00_selayer0_preact_layer1__1_0_se_fc0_linear1_in_features, selayer0_preact_layer1__1_0_se_fc0_linear1_out_features=bottleneck00_selayer0_preact_layer1__1_0_se_fc0_linear1_out_features, selayer0_x5_shape=bottleneck00_selayer0_x5_shape, preact_layer1__1_0_downsample0_conv0_out_channels=bottleneck00_preact_layer1__1_0_downsample0_conv0_out_channels, preact_layer1__1_0_downsample0_conv0_stride=[1, 1], preact_layer1__1_0_downsample0_conv0_in_channels=bottleneck00_preact_layer1__1_0_downsample0_conv0_in_channels, preact_layer1__1_0_downsample0_bn0_num_channels=bottleneck00_preact_layer1__1_0_downsample0_bn0_num_channels)
        self.bottleneck10 = Bottleneck1(conv0_out_channels=bottleneck10_conv0_out_channels, conv0_in_channels=bottleneck10_conv0_in_channels, bn0_num_channels=bottleneck10_bn0_num_channels, conv1_out_channels=bottleneck10_conv1_out_channels, conv1_in_channels=bottleneck10_conv1_in_channels, bn1_num_channels=bottleneck10_bn1_num_channels, conv2_out_channels=bottleneck10_conv2_out_channels, conv2_in_channels=bottleneck10_conv2_in_channels, bn2_num_channels=bottleneck10_bn2_num_channels)
        self.bottleneck11 = Bottleneck1(conv0_out_channels=bottleneck11_conv0_out_channels, conv0_in_channels=bottleneck11_conv0_in_channels, bn0_num_channels=bottleneck11_bn0_num_channels, conv1_out_channels=bottleneck11_conv1_out_channels, conv1_in_channels=bottleneck11_conv1_in_channels, bn1_num_channels=bottleneck11_bn1_num_channels, conv2_out_channels=bottleneck11_conv2_out_channels, conv2_in_channels=bottleneck11_conv2_in_channels, bn2_num_channels=bottleneck11_bn2_num_channels)
    def forward(self, x0):
        x1 = self.bottleneck00(x0)
        x2 = self.bottleneck10(x1)
        x3 = self.bottleneck11(x2)
        return x3

class SEResnet(paddle.nn.Layer):
    def __init__(self, ):
        super(SEResnet, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=64, kernel_size=(7, 7), bias_attr=False, stride=2, padding=3, in_channels=3)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=64, momentum=0.01)
        self.relu0 = paddle.nn.ReLU()
        self.maxpool0 = paddle.nn.MaxPool2D(kernel_size=[3, 3], stride=2, padding=1)
        self.preact_layer1__10 = Preact_layer1__1(bottleneck00_conv0_out_channels=64, bottleneck00_conv0_in_channels=64, bottleneck00_bn0_num_channels=64, bottleneck00_conv1_out_channels=64, bottleneck00_conv1_in_channels=64, bottleneck00_bn1_num_channels=64, bottleneck00_conv2_out_channels=256, bottleneck00_conv2_in_channels=64, bottleneck00_bn2_num_channels=256, bottleneck00_selayer0_x3_shape=[1, 256], bottleneck00_selayer0_preact_layer1__1_0_se_fc0_linear0_in_features=256, bottleneck00_selayer0_preact_layer1__1_0_se_fc0_linear0_out_features=256, bottleneck00_selayer0_preact_layer1__1_0_se_fc0_linear1_in_features=256, bottleneck00_selayer0_preact_layer1__1_0_se_fc0_linear1_out_features=256, bottleneck00_selayer0_x5_shape=[1, 256, 1, 1], bottleneck00_preact_layer1__1_0_downsample0_conv0_out_channels=256, bottleneck00_preact_layer1__1_0_downsample0_conv0_in_channels=64, bottleneck00_preact_layer1__1_0_downsample0_bn0_num_channels=256, bottleneck10_conv0_out_channels=64, bottleneck10_conv0_in_channels=256, bottleneck10_bn0_num_channels=64, bottleneck10_conv1_out_channels=64, bottleneck10_conv1_in_channels=64, bottleneck10_bn1_num_channels=64, bottleneck10_conv2_out_channels=256, bottleneck10_conv2_in_channels=64, bottleneck10_bn2_num_channels=256, bottleneck11_conv0_out_channels=64, bottleneck11_conv0_in_channels=256, bottleneck11_bn0_num_channels=64, bottleneck11_conv1_out_channels=64, bottleneck11_conv1_in_channels=64, bottleneck11_bn1_num_channels=64, bottleneck11_conv2_out_channels=256, bottleneck11_conv2_in_channels=64, bottleneck11_bn2_num_channels=256)
        self.preact_layer20 = Preact_layer2()
        self.preact_layer30 = Preact_layer3()
        self.preact_layer1__11 = Preact_layer1__1(bottleneck00_conv0_out_channels=512, bottleneck00_conv0_in_channels=1024, bottleneck00_bn0_num_channels=512, bottleneck00_conv1_out_channels=512, bottleneck00_conv1_in_channels=512, bottleneck00_bn1_num_channels=512, bottleneck00_conv2_out_channels=2048, bottleneck00_conv2_in_channels=512, bottleneck00_bn2_num_channels=2048, bottleneck00_selayer0_x3_shape=[1, 2048], bottleneck00_selayer0_preact_layer1__1_0_se_fc0_linear0_in_features=2048, bottleneck00_selayer0_preact_layer1__1_0_se_fc0_linear0_out_features=2048, bottleneck00_selayer0_preact_layer1__1_0_se_fc0_linear1_in_features=2048, bottleneck00_selayer0_preact_layer1__1_0_se_fc0_linear1_out_features=2048, bottleneck00_selayer0_x5_shape=[1, 2048, 1, 1], bottleneck00_preact_layer1__1_0_downsample0_conv0_out_channels=2048, bottleneck00_preact_layer1__1_0_downsample0_conv0_in_channels=1024, bottleneck00_preact_layer1__1_0_downsample0_bn0_num_channels=2048, bottleneck10_conv0_out_channels=512, bottleneck10_conv0_in_channels=2048, bottleneck10_bn0_num_channels=512, bottleneck10_conv1_out_channels=512, bottleneck10_conv1_in_channels=512, bottleneck10_bn1_num_channels=512, bottleneck10_conv2_out_channels=2048, bottleneck10_conv2_in_channels=512, bottleneck10_bn2_num_channels=2048, bottleneck11_conv0_out_channels=512, bottleneck11_conv0_in_channels=2048, bottleneck11_bn0_num_channels=512, bottleneck11_conv1_out_channels=512, bottleneck11_conv1_in_channels=512, bottleneck11_bn1_num_channels=512, bottleneck11_conv2_out_channels=2048, bottleneck11_conv2_in_channels=512, bottleneck11_bn2_num_channels=2048)
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x5 = self.maxpool0(x3)
        x6 = self.preact_layer1__10(x5)
        x7 = self.preact_layer20(x6)
        x8 = self.preact_layer30(x7)
        x9 = self.preact_layer1__11(x8)
        return x9

class DUC0(paddle.nn.Layer):
    def __init__(self, ):
        super(DUC0, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=1024, kernel_size=(3, 3), bias_attr=False, padding=1, in_channels=512)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=1024, momentum=0.1)
        self.relu0 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = paddle.nn.functional.pixel_shuffle(x=x3, upscale_factor=2)
        return x4

class DUC1(paddle.nn.Layer):
    def __init__(self, ):
        super(DUC1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=512, kernel_size=(3, 3), bias_attr=False, padding=1, in_channels=256)
        self.bn0 = paddle.nn.BatchNorm(is_test=True, num_channels=512, momentum=0.1)
        self.relu0 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = paddle.nn.functional.pixel_shuffle(x=x3, upscale_factor=4)
        return x4

class FastPose(paddle.nn.Layer):
    def __init__(self, ):
        super(FastPose, self).__init__()
        self.seresnet0 = SEResnet()
        self.duc00 = DUC0()
        self.duc10 = DUC1()
        self.conv0 = paddle.nn.Conv2D(out_channels=2, kernel_size=(3, 3), padding=1, in_channels=32)
    def forward(self, x0):
        x1 = self.seresnet0(x0)
        x2 = paddle.nn.functional.pixel_shuffle(x=x1, upscale_factor=2)
        x3 = self.duc00(x2)
        x4 = self.duc10(x3)
        x5 = self.conv0(x4)
        return x5

def main(x0):
    # There are 1 inputs.
    # x0: shape-[1, 3, 224, 224], type-float32.
    paddle.disable_static()
    params = paddle.load('weight/model.pdparams')
    model = FastPose()
    model.set_dict(params)
    paddle.save(model.state_dict(), 'weight/SEResnte')
    model.eval()
    out = model(x0)
    return out
# import numpy as np
# main(paddle.to_tensor(np.ones((2,3,512,512),np.float32)))