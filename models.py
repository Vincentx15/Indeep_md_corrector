import torch


class Corrector(torch.nn.Module):
    """GCN model that iteratively applies edge contraction and computes node embeddings.
    """

    def __init__(self, mid_channel=16):
        """
        :param n_features: number of input features (default=16)
        """
        super(Corrector, self).__init__()
        self.layer1_1 = torch.nn.Conv1d(in_channels=2,
                                        out_channels=mid_channel,
                                        kernel_size=11,
                                        dilation=1,
                                        bias=True,
                                        padding='same',
                                        padding_mode='reflect')
        self.layer1_2 = torch.nn.Conv1d(in_channels=2,
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
        out_1 = self.layer1_1(x)
        out_2 = self.layer1_2(x)
        mid_out = torch.cat((out_1, out_2), dim=1)
        out = self.layer2(mid_out)
        return out
