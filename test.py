import argparse
import logging
import os

import torch
import torch.utils.data

from ssd.config import cfg
from ssd.engine.inference import do_evaluation
from ssd.modeling.detector import build_detection_model
from ssd.utils import dist_util
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.dist_util import synchronize
from ssd.utils.logger import setup_logger


def evaluation(cfg, ckpt, distributed,model_path=None):
    logger = logging.getLogger("SSD.inference")
    model = build_detection_model(cfg)
    logger.info("Model :\n{}".format(model))#如果用print,多gpu会打印两便
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR, logger=logger)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    if model_path is None:
        checkpointer.load(ckpt, use_latest=ckpt is None)
    else:
        model.load_state_dict(torch.load(model_path))
    if cfg.TEST.BN_FUSE is True:
        print('BN_FUSE.')
        model.backbone.bn_fuse()
        model.to(device)
    do_evaluation(cfg, model, distributed)


def main():
    parser = argparse.ArgumentParser(description='SSD Evaluation on VOC and COCO dataset.')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--model_path",
        help="The path to the model for test, only model",
        default=None,
        type=str,
    )


    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    #print("Test only use one gpu!")#多gpu出错，待改进


    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if num_gpus>1:
        exit()
    distributed = num_gpus > 1

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = setup_logger("SSD", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    evaluation(cfg, ckpt=args.ckpt, distributed=distributed,model_path=args.model_path)


if __name__ == '__main__':
    main()
