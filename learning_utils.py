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


if __name__ == "__main__":
    loss = RbfLoss(min_value=0, max_value=4, nbins=2)
    gt = torch.linspace(1, 3, steps=1)
    pred1 = gt
    pred2 = gt + 0.1
    pred3 = gt + 10

    gt = gt[:, None]
    pred1 = pred1[:, None]
    pred2 = pred2[:, None]
    pred3 = pred3[:, None]
    loss_value_1 = loss(pred1, gt)
    loss_value_2 = loss(pred2, gt)
    loss_value_3 = loss(pred3, gt)
    print(loss_value_1, loss_value_2, loss_value_3)
