import numpy as np
from scipy.stats.stats import pearsonr
import sys
import torch
import matplotlib.pyplot as plt

import loader
import models
from utils import clonumpy


def one_call(data, model, criterion, compute_correlation=True):
    loss = []
    correlations = []
    for df in data:
        data_in, data_out = loader.process_df(df)
        out = model(data_in)
        loss.append(criterion(out, data_out))
        if compute_correlation:
            np_pred, np_out = clonumpy(out), clonumpy(data_out)
            correlations.append(pearsonr(np_pred, np_out))
    loss = torch.mean(torch.stack(loss))
    correlations = np.mean(correlations)
    return loss, correlations


if __name__ == '__main__':
    pass
    torch.random.manual_seed(42)
    META = False
    CORRECT_MODE = True
    IN_CHANNELS = 2
    MID_CHANNELS = 16
    data_train = loader.load_all(meta=META)
    data_test = loader.load_all(train=False, meta=META)
    # model = models.Corrector(mid_channel=MID_CHANNELS, in_channels=IN_CHANNELS, correct_mode=CORRECT_MODE)
    model = models.DoubleCorrector(mid_channel=MID_CHANNELS)
    criterion = torch.nn.MSELoss()

    # from cProfile import Profile
    # from pstats import Stats
    # profiler = Profile()
    # profiler.runcall(lambda: one_call(data=data_train, model=model, criterion=criterion))
    # stats = Stats(profiler)
    # stats.strip_dirs()
    # stats.sort_stats('cumulative')
    # stats.print_callers()
    # sys.exit()

    # Baseline :
    baseline_train_loss, baseline_train_correlation = one_call(data=data_train,
                                                               model=lambda x: x[:, 0:1],
                                                               criterion=criterion)
    print(f"Training baseline loss : {baseline_train_loss.item()} and correlation {baseline_train_correlation}")

    baseline_test_loss, baseline_test_correlation = one_call(data=data_test,
                                                             model=lambda x: x[:, 0:1],
                                                             criterion=criterion)
    print(f"Testing baseline loss : {baseline_test_loss.item()} and correlation {baseline_test_correlation}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs = 200
    all_corrs_train = list()
    all_corrs_test = list()
    for i, epoch in enumerate(range(epochs)):
        optimizer.zero_grad()
        train_loss, train_correlation = one_call(data=data_train, model=model, criterion=criterion)
        train_loss.backward()
        optimizer.step()
        if not i % 1:
            with torch.no_grad():
                test_loss, test_correlation = one_call(data=data_test, model=model, criterion=criterion)
            print(f"After {i} iterations, Training loss : {train_loss.item()}, Training corr : {train_correlation}, "
                  f"Test loss, {test_loss.item()}, test corr : {test_correlation}")
            all_corrs_train.append(train_correlation)
            all_corrs_test.append(test_correlation)

    plt.plot(all_corrs_train, label='train')
    plt.plot(all_corrs_test, label='test')
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()
    # name = f"{'meta' if META else 'md'}_{IN_CHANNELS}_{'correct' if CORRECT_MODE else 'direct'}_corrector.pt"
    name = "double_corrector.pt"
    torch.save(model, name)
