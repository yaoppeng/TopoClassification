import wandb
import torch
import numpy as np
import click
import ast


class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


def initialize_wandb(name="Skin_lesion_project", project="Skin_lesion_latest",
                     dir="wandb", id=None):
    run = wandb.init(
        project=project,
        dir=dir,
        name=name,
        resume="allow",  # must resume, otherwise crash
        id=id
    )

    return run


def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def sen(n, con_mat):  # n为分类数

    sen = []
    # con_mat = confusion_matrix(Y_test, Y_pred)

    if n == 2:
        for i in range(1, n):
            tp = con_mat[i][i]
            fn = np.sum(con_mat[i, :]) - tp
            sen1 = tp / (tp + fn)
            sen.append(sen1)

    else:
        for i in range(n):
            tp = con_mat[i][i]
            fn = np.sum(con_mat[i, :]) - tp
            sen1 = tp / (tp + fn)
            sen.append(sen1)

    return sen


def pre(n, con_mat):
    pre = []
    # con_mat = confusion_matrix(Y_test, Y_pred)

    if n == 2:
        for i in range(1, n):
            tp = con_mat[i][i]
            fp = np.sum(con_mat[:, i]) - tp
            pre1 = tp / (tp + fp)
            pre.append(pre1)

    else:
        for i in range(n):
            tp = con_mat[i][i]
            fp = np.sum(con_mat[:, i]) - tp
            pre1 = tp / (tp + fp)
            pre.append(pre1)

    return pre


def spe(n, con_mat):
    spe = []
    # con_mat = confusion_matrix(Y_test, Y_pred)

    if n == 2:
        for i in range(1, n):
            number = np.sum(con_mat[:, :])
            tp = con_mat[i][i]
            fn = np.sum(con_mat[i, :]) - tp
            fp = np.sum(con_mat[:, i]) - tp
            tn = number - tp - fn - fp
            spe1 = tn / (tn + fp)
            spe.append(spe1)
    else:
        for i in range(n):
            number = np.sum(con_mat[:, :])
            tp = con_mat[i][i]
            fn = np.sum(con_mat[i, :]) - tp
            fp = np.sum(con_mat[:, i]) - tp
            tn = number - tp - fn - fp
            spe1 = tn / (tn + fp)
            spe.append(spe1)

    return spe


def ACC(n, con_mat):
    acc = []
    # con_mat = confusion_matrix(Y_test, Y_pred)
    if n == 2:
        for i in range(1, n):
            number = np.sum(con_mat[:, :])
            tp = con_mat[i][i]
            fn = np.sum(con_mat[i, :]) - tp
            fp = np.sum(con_mat[:, i]) - tp
            tn = number - tp - fn - fp
            acc1 = (tp + tn) / number
            acc.append(acc1)
    else:
        for i in range(n):
            number = np.sum(con_mat[:, :])
            tp = con_mat[i][i]
            fn = np.sum(con_mat[i, :]) - tp
            fp = np.sum(con_mat[:, i]) - tp
            tn = number - tp - fn - fp
            acc1 = (tp + tn) / number
            acc.append(acc1)

    return acc


def prepare_input(resolution):
    x1 = torch.FloatTensor(1, 3, 224, 224)
    x2 = None  # torch.FloatTensor(1, 4, 200)
    x3 = None  # torch.FloatTensor(1, 16, 500)

    return dict(inputs=(x1, x2, x3))  # dict(x = [x1, x2])


def get_current_lr(epoch):
    return 0.9 ** epoch


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x = [i for i in range(1000)]
    y = [1 * get_current_lr(i) for i in range(1000)]
    plt.plot(x, y); plt.show()
    print(f"finishe")