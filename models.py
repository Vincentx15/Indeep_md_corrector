import torch
from utils import clonumpy


class Corrector(torch.nn.Module):
    def __init__(self, in_channels=2, mid_channel=16, correct_mode=False):
        """
        :param n_features: number of middle features (default=16)
        """
        super(Corrector, self).__init__()
        self.in_channels = in_channels
        self.correct_mode = correct_mode
        self.layer1_1 = torch.nn.Conv1d(in_channels=in_channels,
                                        out_channels=mid_channel,
                                        kernel_size=11,
                                        dilation=1,
                                        bias=True,
                                        padding='same',
                                        padding_mode='reflect')
        self.layer1_2 = torch.nn.Conv1d(in_channels=in_channels,
                                        out_channels=mid_channel,
                                        kernel_size=7,
                                        dilation=2,
                                        bias=True,
                                        padding='same',
                                        padding_mode='reflect')
        self.layer2 = torch.nn.Conv1d(in_channels=2 * mid_channel,
                                      out_channels=1,
                                      kernel_size=5,
                                      stride=1,
                                      dilation=1,
                                      bias=True,
                                      padding='same',
                                      padding_mode='reflect')

    def forward(self, x):
        """
        :param x:
        :return:
        """
        original_ligandabilities = x[:, 0:1]
        # If we want to just learn correction
        if self.in_channels == 1 and x.shape[1] > 1:
            x = original_ligandabilities
        out_1 = self.layer1_1(x)
        out_2 = self.layer1_2(x)
        mid_out = torch.cat((out_1, out_2), dim=1)
        out = self.layer2(mid_out)

        # print(torch.sigmoid(out)[0, 0, :40])
        # post_sig = clonumpy(torch.sigmoid(out)[0, 0, :])
        # import matplotlib.pyplot as plt
        # plt.plot(post_sig, alpha=0.5, label='old')
        # plt.show()

        if not self.correct_mode:
            return out
        return - torch.sigmoid(out) * original_ligandabilities


class DoubleCorrector(torch.nn.Module):
    def __init__(self, mid_channel=16):
        """
        :param n_features: number of middle features (default=16)
        """
        super(DoubleCorrector, self).__init__()
        self.corrector = Corrector(mid_channel=mid_channel, in_channels=2, correct_mode=False)
        self.smoother = Corrector(mid_channel=mid_channel, in_channels=1, correct_mode=False)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        original_ligandabilities = x[:, 0:1]
        smoothed_ligandabilities = self.smoother(original_ligandabilities)
        correction = self.corrector(x)
        return torch.sigmoid(correction) * smoothed_ligandabilities
