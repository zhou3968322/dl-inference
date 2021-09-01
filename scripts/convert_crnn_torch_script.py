# -*- coding:utf-8 -*-
# email:bingchengzhou@foxmail.com
# create: 2021/3/19

import torch.nn as nn
import torch.nn.functional as F


class BidirectionalLSTM(nn.Module):

    def __init__(self, n_in, n_hidden, n_out):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(n_in, n_hidden, bidirectional=True)
        self.embedding = nn.Linear(n_hidden * 2, n_out)

    def forward(self, inp):
        recurrent, _ = self.rnn(inp)  # [T, b, h * 2]

        t, b, h = recurrent.size()
        t_rec = recurrent.view(t * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(t, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, img_h, n_c, n_class, n_h, is_leaky_relu=False):
        super(CRNN, self).__init__()
        assert img_h % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, (img_h // 16, 2)]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def conv_relu(i, batch_normalization=False):
            n_in = n_c if i == 0 else nm[i - 1]
            n_out = nm[i]
            cnn.add_module('conv{0}'.format(i), nn.Conv2d(n_in, n_out, ks[i], ss[i], ps[i]))
            if batch_normalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(n_out))
            if is_leaky_relu:
                cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        conv_relu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        conv_relu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        conv_relu(2, True)
        conv_relu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        conv_relu(4, True)
        conv_relu(5)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        conv_relu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(BidirectionalLSTM(512, n_h, n_h), BidirectionalLSTM(n_h, n_h, n_class))

    def forward(self, inp):
        # conv features
        conv = self.cnn(inp)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)
        # output = F.log_softmax(output, dim=2)
        return output


def convert_crnn_jit():
    import torch
    crnn_model = CRNN(48, 1, 7500, 256)
    model_path = "/data/duser/models/ysocr_models/recognition_models/document/document.pth"
    device = torch.device("cuda:0")
    crnn_model.load_state_dict(torch.load(model_path, map_location=device))
    crnn_model = crnn_model.to(device)
    sm = torch.jit.script(crnn_model)
    sm_model_path = "/data/duser/models/ysocr_models/recognition_models/document/jit_document.pt"
    sm.save(sm_model_path)


if __name__ == '__main__':
    convert_crnn_jit()