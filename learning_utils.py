import torch


class RbfLoss:
    """
    Loss to map a continuous value to a set of RBF sensors to turn regression into classification
    """

    def __init__(self, min_value, max_value, nbins):
        self.centers = torch.linspace(min_value, max_value, nbins)[:, None]
        self.std = (max_value - min_value) / (2 * nbins)
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def __call__(self, prediction, ground_truth):
        dists_pred = torch.cdist(prediction, self.centers)
        dists_gt = torch.cdist(ground_truth, self.centers)

        smoothed_preds = torch.exp(-dists_pred / self.std)
        # smoothed_preds = torch.nn.Softmax(dim=1)(-dists_pred / self.std)
        smoothed_labels = torch.nn.Softmax(dim=1)(-dists_gt / self.std)

        loss = self.cross_entropy(smoothed_preds, smoothed_labels)
        return loss

    def to(self, device):
        self.centers = self.centers.to(device)
        return self


def categorical_loss(prediction, ground_truth, bins=(1, 1.5, 2)):
    """
    <1 Ang, entre 1 et 1.5, entre 1.5 et 2, et plus de 2
    :param prediction:
    :param ground_truth:
    :param bins:
    :return:
    """
    # First let's bin the gt :
    binned_gt = [ground_truth < bins[0]]
    for i, (low, high) in enumerate(zip(bins, bins[1:])):
        binned_gt.append(torch.logical_and(low <= ground_truth, ground_truth < high))
    binned_gt.append(bins[-1] <= ground_truth)
    binned_gt = torch.concat(binned_gt, 1).int()
    targets = torch.argmax(binned_gt, dim=1)
    return torch.nn.CrossEntropyLoss()(prediction, targets)


if __name__ == "__main__":
    # loss = RbfLoss(min_value=0, max_value=4, nbins=2)
    # gt = torch.linspace(0, 3, steps=5)
    # pred1 = gt
    # pred2 = gt + 0.1
    # pred3 = gt + 10
    # gt = gt[:, None]
    # pred1 = pred1[:, None]
    # pred2 = pred2[:, None]
    # pred3 = pred3[:, None]
    # loss_value_1 = loss(pred1, gt)
    # loss_value_2 = loss(pred2, gt)
    # loss_value_3 = loss(pred3, gt)
    # print(loss_value_1, loss_value_2, loss_value_3)

    bins = (1, 1.5, 2)

    gt = torch.linspace(0, 3, steps=5)[:, None]
    pred = torch.randn(size=(len(gt), len(bins) + 1))
    categorical_loss(pred, gt)
