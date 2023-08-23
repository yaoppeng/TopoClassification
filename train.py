import time
from models.swin_utils import *
from models import model
from torch.utils.data.dataloader import DataLoader
from dataset.isic_dataset import ISICDataset
# from models.swin_transformer import swin_b, swin_l, swin_t, swin_s
# from torchvision.models.swin_transformer import swin_b, swin_v2_b
from models.swin_transformer_v2 import TopoSwinTransformer, swin_v2_b
from dataset.mammography_dataset import MammographyDataset
import sys
import cv2
from dataset.prostate_dataset import ProstateDataset
from dataset.glaucoma_dataset import GlaucomaDataset
from utils.lr_scheduler import PolyLRScheduler
import click
import os
from batchgenerators.utilities.file_and_folder_operations import *
from pynvml import *
from models.swin_transformer import SwinTransformer
from glob import glob

sys.path.append(join(os.getenv('HOME'), 'TopoClassification'))

import sys
import torch.nn as nn
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from utils.function import get_lr
from utils.metric import mean_class_recall
from utils.printer import Printer
from utils.my_utils import PythonLiteralOption
from utils.my_utils import *
# from ptflops import get_model_complexity_info
from thop import profile
from utils.seed import seed_torch


@click.command()
@click.option("--backbone", type=click.Choice(['resnet152', 'resnet50',
                                               'senet154', 'res2net', 'SwinT']))
@click.option("--cluster_id", default="default_id", type=str)
@click.option("--topo_layers", cls=PythonLiteralOption)
@click.option("--use_ph", type=bool, default=True)
@click.option("--n_epochs", type=int, default=1000)
@click.option("--debug", type=bool, default=True)
@click.option("--ph_type", type=click.Choice(['concate', 'distill', 'phg']),
              default="phg")
@click.option("--use_cnn", type=bool, default=True)
@click.option("--alpha", type=float)
@click.option("--resume", type=bool, default=False)
@click.option("--topo_model", type=click.Choice(['perslay', 'pllay', 'pointnet']),
              default="pointnet")
@click.option("--fold", type=int, default=2)
@click.option("--n_classes", type=int)
@click.option("--optimizer", type=click.Choice(['adam', 'sgd']))
@click.option("--loss_fn", type=click.Choice(['CE', 'WCE']), default="WCE")
@click.option("--eval_frequency", type=int, default=10)
@click.option("--learning_rate", type=float, default=1e-3)
@click.option("--batch_size", type=int, default=16)
@click.option("--round", type=int, default=0)
@click.option("--data_base", type=click.Choice(['Prostate', 'Glaucoma', 'CBIS_DDSM', 'ISIC']),
              default='CBIS_DDSM')
@click.option("--sub_type", type=click.Choice(['all', 'calc', 'mass']))
@click.option("--wandb_key", type=str)
def main(**configs_dict):
    seed_torch()
    debug = configs_dict['debug']
    data_base = configs_dict['data_base']
    sub_type = configs_dict['sub_type']
    alpha = configs_dict['alpha']
    wandb_key = configs_dict['wandb_key']
    if not debug:
        wandb.login(key=wandb_key)
    topo_layers = configs_dict['topo_layers']
    cluster_id = configs_dict['cluster_id']

    """
    config      use_ph      use_cnn
    
                  Y           N
                  N           Y
                  Y           Y     
    """
    use_ph = configs_dict['use_ph']
    use_cnn = configs_dict['use_cnn']

    n_epochs = configs_dict["n_epochs"]
    round = configs_dict['round']
    batch_size = configs_dict["batch_size"]
    learning_rate = configs_dict["learning_rate"]
    backbone = configs_dict["backbone"]
    eval_frequency = configs_dict["eval_frequency"]
    resume = configs_dict["resume"]
    optimizer = configs_dict["optimizer"]
    # initialization = configs_dict["initialization"]
    num_classes = configs_dict["n_classes"]
    fold = configs_dict["fold"]
    loss_fn = configs_dict["loss_fn"]
    ph_type = configs_dict['ph_type']
    topo_model = configs_dict['topo_model']

    # home_dir = join(os.getenv('HOME'), "TopoClassification", 'trained_model')
    home_dir = join(os.getenv('HOME'), "TopoClassification", 'trained_model')

    if 'CBIS_DDSM' == data_base:
        base_save_dir = join(home_dir, data_base, sub_type, f"round_{round}", f"fold_{fold}")
    else:
        base_save_dir = join(home_dir, data_base, f"round_{round}", f"fold_{fold}")
    """
    config      use_ph      use_cnn
                  Y           N
                  N           Y
                  Y           Y     
    """

    if use_ph and not use_cnn:
        # only use topology, no CNN
        log_base = join("only_topo", topo_model, "logs")
        model_base = join("only_topo", topo_model, "models")
        wandb_base = join("only_topo", topo_model)
    elif not use_ph and use_cnn:
        # only use CNN
        log_base = join("only_cnn", backbone, "logs")
        model_base = join("only_cnn", backbone, "models")
        wandb_base = join("only_cnn", backbone)
    elif use_ph and use_cnn:
        # CNN + topology
        log_base = join("both", backbone, topo_model, ph_type, "logs")
        model_base = join("both", backbone, topo_model, ph_type, "models")
        wandb_base = join("both", backbone, topo_model, ph_type)
    else:
        raise ValueError(f"Invalid config: use_ph {use_ph}, use_cnn {use_cnn}")

    log_dir = join(base_save_dir, log_base)
    model_dir = join(base_save_dir, model_base)
    wandb_dir = join(base_save_dir, wandb_base)

    printer = Printer(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # init_environment(seed=seed, cuda_id=cuda_ids)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    """
    mean and std:
        Prostate: tensor([0.6576, 0.4719, 0.6153]) tensor([0.2117, 0.2266, 0.1821])
    """

    mean_std = {
        "Prostate": {"mean": [0.6576, 0.4719, 0.6153], "std": [0.2117, 0.2266, 0.1821]},
        "Glaucoma": {"mean": [0.7266, 0.3759, 0.0983], "std": [0.1806, 0.1580, 0.0861]},
        "CBIS_DDSM": {
            "all": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            # "calc": {"mean": [0.6941, 0.6941, 0.6941], "std": [0.1769, 0.1769, 0.1769]},
            "calc": {"mean": [0.5], "std": [0.5]},
            "mass": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        },
        "ISIC": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    }
    if backbone in ["resnet50", "resnet18", "senet154", "resnet152", "SwinT"]:
        re_size = 300
        input_size = 224

        if data_base == 'CBIS_DDSM':
            mean = mean_std[data_base][sub_type]['mean']
            std = mean_std[data_base][sub_type]['std']
        else:
            mean = mean_std[data_base]['mean']
            std = mean_std[data_base]['std']

    else:
        printer.print_to_log_file("Need backbone")
        sys.exit(-1)
    # from monai.networks.nets.densenet import DenseNet121
    # from monai.networks.nets.resnet import ResNet
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.ToPILImage(),
        # transforms.Resize(re_size),
        transforms.Normalize(mean=mean, std=std),
        # transforms.Resize(re_size),
        transforms.Resize((input_size, input_size), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.02, 0.02, 0.02, 0.01),
        transforms.RandomRotation([-180, 180]),
        transforms.RandomAffine([-180, 180], translate=[0.1, 0.1],
                                scale=[0.7, 1.3]),
        # transforms.RandomCrop(input_size),

    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.ToPILImage(),
        transforms.Normalize(mean, std),
        transforms.Resize((input_size, input_size), antialias=True),
        # transforms.ToTensor(),
    ])

    home_dir = join(os.getenv('HOME'), 'data/raw_data')
    # dataset_paths = {'Prostate': join(home_dir, 'Prostate/rois'),
    #                  'Retina': None}
    dataset_paths = {'Prostate': join(home_dir, 'Prostate/rois'),
                     'Glaucoma': join(home_dir, 'Glaucoma/Glaucoma'),
                     'CBIS_DDSM': join(home_dir, 'CBIS_DDSM'),
                     "ISIC": join(home_dir, "ISIC2018")}
    label_files = {
        'Prostate': join(home_dir, "Prostate/rois-with-grade.csv"),
        'Glaucoma': join(home_dir, "Glaucoma/label.csv"),
        'CBIS_DDSM': {
            'all': join(home_dir, "CBIS_DDSM/all_cases.csv"),
            'calc': join(home_dir, "CBIS_DDSM/calc.csv"),
            'mass': join(home_dir, "CBIS_DDSM/mass.csv"),
        },
        "ISIC": join(home_dir, "ISIC/label.csv")
    }

    data_path = dataset_paths[data_base]
    if 'CBIS_DDSM' == data_base:
        label_file = label_files[data_base][sub_type]
    else:
        label_file = label_files[data_base]

    # split = load_pickle(join(home_dir, "Prostate", "splits_5_fold.pkl"))
    # split = load_json(join(home_dir, data, f"splits_3_fold.json"))

    datasets = {
        "Prostate": ProstateDataset,
        "Glaucoma": GlaucomaDataset,
        "CBIS_DDSM": MammographyDataset,
        "ISIC": ISICDataset
    }

    train_set = datasets[data_base](
        img_path=data_path,
        label_file=label_file,
        transform=train_transform, fold=fold, train=True
    )
    # from torchvision.models.swin_transformer.S
    val_set = datasets[data_base](
        img_path=data_path,
        label_file=label_file,
        transform=val_transform, fold=fold, train=False
    )
    topo_model = "pointnet"
    import inspect
    from torchvision.models.swin_transformer import SwinTransformer
    # print(list(inspect.signature(swin_b).parameters))
    """
    patch_size: List[int],
    embed_dim: int,
    depths: List[int],
    num_heads: List[int],
    window_size: List[int],
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
    attention_dropout: float = 0.0,
    stochastic_depth_prob: float = 0.1,
    num_classes: int = 1000,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    block: Optional[Callable[..., nn.Module]] = None,
    downsample_layer: Callable[..., nn.Module] = PatchMerging,
    """
    input_channel = 1
    use_cnn = True
    ph_type = "concate"
    if use_ph and use_cnn:
        if backbone == "SwinT":
            net = TopoSwinTransformer(
                patch_size=[4, 4],
                embed_dim=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                window_size=[8, 8],
                stochastic_depth_prob=0.5,
                block=SwinTransformerBlockV2,
                downsample_layer=PatchMergingV2,
                num_classes=1000,
                topo_setting=0,  # using pointnet,
                topo_layers=topo_layers
            ).cuda()

            # net = swin_v2_b(num_classes=1000).cuda()
            # net = swin_v2_b(weights="IMAGENET1K_V1", num_classes=1000).cuda()
            """
                net = swin_v2_b(weights="IMAGENET1K_V1", num_classes=1000).cuda()
                # net = swin_b(weights="IMAGENET1K_V1", num_classes=1000).cuda()
                pretrained_weights = net.features[0][0].weight.data
                net.features[0][0] = nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))
                # nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
                net.features[0][0].weight.data = pretrained_weights[:, :1, :, :]
            """
            net.head = nn.Linear(in_features=1024, out_features=num_classes, bias=True).cuda()
            net.topo_linear = nn.Linear(in_features=2048, out_features=num_classes).cuda()
        else:
            net = model.Network(backbone=backbone, num_classes=num_classes,
                                input_channel=input_channel, pretrained="pretrained",
                                use_topology=use_ph, topo_layers=topo_layers,
                                share_topo=True, he_init=True,
                                topo_model=topo_model,
                                topo_setting=0).cuda()
    elif use_cnn and not use_ph:
        if backbone == "SwinT":
            net = swin_v2_b(weights="IMAGENET1K_V1", num_classes=1000,).cuda()
            net.head = nn.Linear(in_features=1024, out_features=num_classes, bias=True).cuda()
        else:
            net = model.Network(backbone=backbone, num_classes=num_classes,
                                input_channel=input_channel, pretrained="pretrained",
                                use_topology=use_ph, topo_layers={1: False, 2: False, 3: False, 4: False},
                                share_topo=False, he_init=True,
                                topo_model=topo_model, topo_setting=0).cuda()

    printer.print_to_log_file({
        "backbone": backbone,
        "use_topology": use_ph,
        "topo_layers": {1: False,
                        2: False,
                        3: False,
                        4: True},
        "share_topo": False, "he_init": True, "use_cnn": use_cnn,
        "topo_model": topo_model, "concate": ph_type == 'concate'
    })

    if not use_cnn:
        # only topo models
        name = f"{cluster_id}_only_topo_{topo_model}"
    elif use_ph:
        # cnn + topo
        name = f"{cluster_id}_cnn_topo_{backbone}_Layer_4_{topo_model}_" \
               f"{ph_type}_without_he_init_sharetopo_" \
               f"False_padding_without_pc_normalize_without_stn"
    else:
        # only cnn
        name = f"{cluster_id}_only_cnn_{backbone}_swin_B_v2"

    if not debug:
        os.makedirs(wandb_dir, exist_ok=True)
        if 'CBIS_DDS' == data_base:
            initialize_wandb(project=f"{data_base}_{sub_type}_round_{round}_fold_{fold}",
                             name=name, dir=wandb_dir)
        else:
            initialize_wandb(project=f"{data_base}_round_{round}_fold_{fold}",
                             name=name, dir=wandb_dir)

    if not debug:
        wan_id = wandb.run.id
        wan_name = wandb.run.name

        printer.print_to_log_file(f"wandb_id: {wan_id}, wan_name: {wan_name}")

    print(f"model_dir: {model_dir}")

    train_sampler = val_sampler = None

    # pdb.set_trace()
    if torch.cuda.is_available():
        nvmlInit()
        gpu = nvmlDeviceGetHandleByIndex(0)
        gpu_name = nvmlDeviceGetName(gpu)
        total_memory = nvmlDeviceGetMemoryInfo(gpu).total

        # if total_memory / (1024 * 2014 * 1024) < 14:
        #     batch_size = batch_size // 4

        printer.print_to_log_file(
            f"I am using 1 gpus on {gpu_name}, {total_memory / (1024 * 2014 * 1024)} G.")

    printer.print_to_log_file("=> iter_fold is {}".format(fold))

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, pin_memory=True,
                              num_workers=12,
                              sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, pin_memory=True,
                            num_workers=12,
                            sampler=val_sampler)

    loss_weights = {
        "Prostate": [0.08463559, 0.09241155, 0.82295286],
        "Glaucoma": [0.56045519, 0.43954481],
        "CBIS_DDSM": {
            "all": [],
            "calc": [0.64049145, 0.35950855],
            "mass": []

        },
        "ISIC": [0.036, 0.002, 0.084, 0.134, 0.037, 0.391, 0.316]
    }
    if loss_fn == "WCE":
        printer.print_to_log_file("Loss function is WCE")
        if data_base == "CBIS_DDSM":
            weights = loss_weights[data_base][sub_type]
        else:
            weights = loss_weights[data_base]

        class_weights = torch.FloatTensor(weights)
        if torch.cuda.is_available():
            class_weights.cuda()
        criterion_1 = nn.CrossEntropyLoss(weight=class_weights).cuda()
        criterion_2 = nn.CrossEntropyLoss(weight=class_weights).cuda()
    elif loss_fn == "CE":
        printer.print_to_log_file("Loss function is CE")
        criterion_1 = nn.CrossEntropyLoss().to(device)
        criterion_2 = nn.CrossEntropyLoss().to(device)
    else:
        criterion_1 = None
        criterion_2 = None
        printer.print_to_log_file("Need loss function.")
        raise ValueError(f"error!")

    # optimizer
    scheduler = None

    if optimizer == "sgd":
        printer.print_to_log_file("=> Using optimizer SGD with lr:{:.4f}".format(learning_rate))
        opt = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     opt, mode='min', factor=0.1, patience=50, verbose=True,
        #     threshold=1e-4)
    elif optimizer == "adam":
        printer.print_to_log_file("=> Using optimizer Adam with lr:{:.4f}".format(learning_rate))
        opt = torch.optim.Adam(net.parameters(), lr=learning_rate,
                               betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    else:
        printer.print_to_log_file("Need optimizer")
        sys.exit(-1)

    scheduler = PolyLRScheduler(opt, initial_lr=learning_rate, max_steps=n_epochs)

    epoch = 0
    if resume:
        pth_files = glob(join(model_dir, "*.pth"))
        pth_files.sort()
        ckpt = torch.load(pth_files[-1], map_location=device)
        printer.print_to_log_file("=> Resume from models at epoch {}".format(pth_files[-1]))

        net.load_state_dict(ckpt['models'], strict=False)
        # start_epoch = resume + 1

        epoch = ckpt['epoch'] + 1
        opt.load_state_dict(ckpt['optimizer'])
    else:
        printer.print_to_log_file("Train from scrach!!")

    sota = {"epoch": epoch, "acc": -1.0}

    printer.print_to_log_file(net)
    printer.print_to_log_file(f"save to {model_dir}")
    # pdb.set_trace()
    while epoch < n_epochs:
        net.train()
        losses = []
        val_losses = []
        printer.print_to_log_file(f"============== epoch: {epoch} =========================")
        for batch_idx, (data, target, ph, pl) in enumerate(train_loader):
        # for batch_idx, (data, target) in enumerate(train_loader):
            # print(f"batch_idx: {batch_idx}")
            # data, target, ph, pl = data.to(device), target.to(device), ph.to(device), pl.to(device)
            data, target = data.float().to(device), target.to(device)
            if ph is not None:
                ph = ph.float().cuda()
            if pl is not None:
                pl = pl.float().cuda()
            #           (3, 224, 224), (4, 4000), (16, 2500)

            predict, topo = net([data, ph, pl])

            # if not use_cnn:
            #     predict = topo_feature

            opt.zero_grad()
            loss1 = criterion_1(predict, target)
            loss2 = criterion_2(topo, target)
            loss = loss1 + alpha * loss2
            loss.backward()
            opt.step()
            losses.append(loss.item())

            # if debug:
            #     break

        train_avg_loss = np.mean(losses)

        if not debug:
            wandb.log({"loss/train_loss": train_avg_loss, "learning_rate/learning_rate": get_lr(opt)}, step=epoch)
        if scheduler is not None:
            # scheduler.step(train_avg_loss)
            scheduler.step(epoch)

        if epoch % eval_frequency == 0:
            with torch.no_grad():
                net.eval()
                y_true = []
                y_pred = []
                times = []
                for _, (data, target, ph, pl) in enumerate(train_loader):
                # for _, (data, target) in enumerate(train_loader):
                    data = data.float().cuda()
                    target = target.cuda()

                    if ph is not None:
                        ph = ph.float().cuda()
                    if pl is not None:
                        pl = pl.float().cuda()

                    # predict, topo_feature = net([data, ph, pl])
                    # predict = net([data, None, None])
                    start = time.time()
                    predict, topo = net([data, ph, pl])
                    end = time.time()
                    times.append(end-start)
                # if not use_cnn:
                    #     predict = topo_feature

                    predict = torch.argmax(predict, dim=1).cpu().data.numpy()

                    y_pred.extend(predict)
                    target = target.cpu().data.numpy()
                    y_true.extend(target)

                    # if debug:
                    #     break

                acc = accuracy_score(y_true, y_pred)
                mcr = mean_class_recall(y_true, y_pred)

                if not debug:
                    wandb.log({"accuracy/train_acc": acc}, step=epoch)

                printer.print_to_log_file("=> Epoch:{} - train acc: {:.4f}".format(epoch, acc))

                y_true = []
                y_pred = []
                y_prob = []
                for _, (data, target, ph, pl) in enumerate(val_loader):
                # for _, (data, target) in enumerate(val_loader):
                    data = data.float().cuda()
                    target = target.cuda()
                    if ph is not None:
                        ph = ph.float().cuda()
                    if pl is not None:
                        pl = pl.float().cuda()

                    # p, topo_feature = net([data, ph, pl])
                    # p = net([data, None, None])
                    p, topo = net([data, ph, pl])
                    loss1 = criterion_1(p, target)
                    loss2 = criterion_2(topo, target)
                    loss = loss1 + alpha * loss2
                    # if not use_cnn:
                    #     p = topo_feature
                    prob = torch.softmax(p, dim=-1)
                    predict = torch.argmax(prob, dim=1).cpu().data.numpy()

                    y_pred.extend(predict)
                    y_prob.extend(prob.cpu().data.numpy())
                    target = target.cpu().data.numpy()
                    y_true.extend(target)

                    val_losses.append(loss.item())

                    # if debug:
                    #     break

                y_prob = np.vstack(y_prob)

                acc = accuracy_score(y_true, y_pred)
                # print(y_true, y_pred)
                mcr = mean_class_recall(y_true, y_pred)

                # pdb.set_trace()

                print(len(y_true), y_prob.shape)

                if 'CBIS_DDSM' == data_base or 'Glaucoma' == data_base:
                    # CBIS_DDSM / Glaucoma
                    auc_score_ovr = roc_auc_score(y_true, y_prob[:, 1])
                elif 'Prostate' == data_base or 'ISIC' == data_base:
                    # Prostate / ISIC
                    auc_score_ovr = roc_auc_score(y_true, y_prob, multi_class="ovr",
                                                  average='macro')
                else:
                    raise ValueError(f"invalid data_base {data_base}")
                # auc_score_ovo = roc_auc_score(y_true, y_prob, multi_class="ovo")
                cm = confusion_matrix(y_true, y_pred)  # .ravel()

                # ISIC
                # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                specificity = np.mean(spe(num_classes, cm))
                sensitivity = np.mean(sen(num_classes, cm))

                # Prostate
                # specificity = np.mean(spe(3, cm))
                # sensitivity = np.mean(sen(3, cm))

                # Glaucoma/CBIS-DDSM
                # specificity = np.mean(spe(2, cm))
                # sensitivity = np.mean(sen(2, cm))

                if not debug:
                    wandb.log({"loss/val_loss": np.mean(val_losses), "accuracy/val_acc": acc,
                               "val/val_auc_ovr": auc_score_ovr,
                               "val/specificity": specificity,
                               "val/sensitivity": sensitivity,
                               }, step=epoch)

                printer.print_to_log_file(f"Epoch: {epoch}, train loss: {train_avg_loss:.4f}, "
                                          f"val loss: {np.mean(val_losses):.4f}, val acc: {acc:.4f}, "
                                          f"val auc: {auc_score_ovr:.4f}, val sen: {sensitivity:.4f}, "
                                          f"val spe: {specificity:.4f}")

                # val acc
                if acc > sota["acc"]:
                    sota["mcr"] = mcr
                    sota["acc"] = acc
                    sota["epoch"] = epoch
                    sota["auc"] = auc_score_ovr
                    sota["spe"] = specificity
                    sota["sen"] = sensitivity

                    model_path = os.path.join(model_dir, f"{epoch:0>3}.pth")

                    printer.print_to_log_file("=> Save models in {}".format(model_path))

                    save_dict = {
                        'epoch': epoch,
                        'models': net.state_dict(),
                        'opt': opt.state_dict(),
                        # 'loss': loss
                    }
                    torch.save(save_dict, model_path)

                    if not debug:
                        wandb.save(model_path)

        epoch += 1

    printer.print_to_log_file("Finish Training")
    printer.print_to_log_file(f"Best epoch {sota['epoch']} on val acc: {sota['acc']}")

    if not debug:
        wandb.finish()


if __name__ == "__main__":
    main()
