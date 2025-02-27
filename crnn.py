import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        """
        imgH: image height after resizing (e.g., 32)
        nc: number of channels (1 for grayscale)
        nclass: number of classes (alphabet characters + 1 for CTC blank)
        nh: LSTM hidden size
        """
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2,1), (2,1)),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2,1), (2,1)),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.ReLU(True)
        )
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, input):
        # input shape: (batch, nc, H, W)
        conv = self.cnn(input)  # shape: (batch, channels, H', W')
        b, c, h, w = conv.size()
        # Ideally, h == 1; if not, squeeze the height dimension
        conv = conv.squeeze(2)  # shape: (batch, channels, width)
        conv = conv.permute(2, 0, 1)  # shape: (width, batch, channels)
        output = self.rnn(conv)
        return output

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)
    
    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output