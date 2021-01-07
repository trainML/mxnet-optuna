import pickle
import time
import os
import optuna
from types import SimpleNamespace
import sys
import mxnet as mx
from math import log
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.presets.rcnn import (
    FasterRCNNDefaultTrainTransform,
    FasterRCNNDefaultValTransform,
)
from train_faster_rcnn import get_dataset, get_dataloader, train


def objective(trial):
    momentum = trial.suggest_uniform("momentum", 0, 1)
    wd = trial.suggest_loguniform("wd", 1e-5, 100)

    space = dict(
        network="resnet50_v1b",
        dataset="voc",
        save_prefix=f"{os.environ.get('TRAINML_OUTPUT_PATH')}/",
        horovod=False,
        amp=False,
        resume=False,
        start_epoch=0,
        verbose=False,
        custom_model=False,
        kv_store="nccl",
        log_interval=100,
        save_interval=1,
        val_interval=1,
        disable_hybridization=False,
        static_alloc=False,
        seed=233,
        mixup=False,
        norm_layer=None,
        use_fpn=False,
        num_workers=4,
        gpus="0",
        executor_threads=1,
        epochs=1,
        batch_size=2,
        lr=0.001,
        lr_decay=0.1,
        lr_decay_epoch="14,20",
        lr_warmup=-1,
        lr_warmup_factor=1.0 / 3.0,
        rpn_smoothl1_rho=1.0 / 9.0,
        rcnn_smoothl1_rho=1.0,
        momentum=momentum,
        wd=wd
    )

    sys.setrecursionlimit(1100)
    args = SimpleNamespace(**space)
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    ctx = [mx.gpu(int(i)) for i in args.gpus.split(",") if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # training data
    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset, args)

    # network
    kwargs = {}
    module_list = []
    if args.use_fpn:
        module_list.append("fpn")
    if args.norm_layer is not None:
        module_list.append(args.norm_layer)
        if args.norm_layer == "syncbn":
            kwargs["num_devices"] = len(ctx)

    num_gpus = len(ctx)
    net_name = "_".join(("faster_rcnn", *module_list, args.network, args.dataset))

    net = get_model(
        net_name,
        pretrained_base=True,
        per_device_batch_size=args.batch_size // num_gpus,
        **kwargs
    )
    args.save_prefix += net_name

    for param in net.collect_params().values():
        if param._data is not None:
            continue
        param.initialize()
    net.collect_params().reset_ctx(ctx)
    batch_size = args.batch_size
    train_data, val_data = get_dataloader(
        net,
        train_dataset,
        val_dataset,
        FasterRCNNDefaultTrainTransform,
        FasterRCNNDefaultValTransform,
        batch_size,
        len(ctx),
        args,
    )

    # training
    train(net, train_data, val_data, eval_metric, batch_size, ctx, args)
    name, values = eval_metric.get()
    idx = name.index("mAP")

    trial.set_user_attr('mAP', values[idx])

    return 1 - values[idx]


if __name__ == "__main__":
    study = optuna.load_study(study_name="mxnet_pascal_voc_1", storage=os.environ.get("DB_CONNECTION_STRING"))
    study.optimize(objective, n_trials=50,gc_after_trial=True)