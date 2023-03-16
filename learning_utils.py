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
        pred_dists = torch.cdist(prediction, self.centers) / self.std
        gt_dists = torch.cdist(ground_truth, self.centers) / self.std
        smoothed_preds = torch.nn.Softmax(dim=1)(-pred_dists)
        smoothed_labels = torch.nn.Softmax(dim=1)(-gt_dists)
        loss = self.cross_entropy(smoothed_preds, smoothed_labels)
        return loss


if __name__ == "__main__":
    loss = RbfLoss(min_value=0, max_value=4, nbins=20)
    pred = torch.linspace(1, 3, 10)
    gt = torch.linspace(1, 3, 10) - 0.1

    pred = pred[:, None]
    gt = gt[:, None]
    loss_value = loss(pred, gt)
    print(loss_value)
