import os
import argparse
from clip_reid.config import cfg
from clip_reid.model.make_model_clipreid import make_model

# from clip_reid.datasets.make_dataloader_clipreid import make_dataloader

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import cv2
import glob

# from clip_reid.datasets.dukemtmcreid import DukeMTMCreID
# from clip_reid.datasets.market1501 import Market1501
# from clip_reid.datasets.msmt17 import MSMT17
# from clip_reid.datasets.occ_duke import OCC_DukeMTMCreID
# from clip_reid.datasets.vehicleid import VehicleID
# from clip_reid.datasets.veri import VeRi


# __factory = {
#     "market1501": Market1501,
#     "dukemtmc": DukeMTMCreID,
#     "msmt17": MSMT17,
#     "occ_duke": OCC_DukeMTMCreID,
#     "veri": VeRi,
#     "VehicleID": VehicleID,
# }


# def get_info(cfg):
#     # dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
#     dataset = DukeMTMCreID

#     num_classes = dataset.num_train_pids
#     cam_num = dataset.num_train_cams
#     view_num = dataset.num_train_vids

#     return len(dataset.query), num_classes, cam_num, view_num


def set_img(cfg, img):
    val_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )

    transformed_img = val_transforms(img).unsqueeze(0)

    return transformed_img


def do_inference(cfg, model, img, camids=None, target_view=None):
    device = "cuda"
    # if device:
    #     if torch.cuda.device_count() > 1:
    #         print("Using {} GPUs for inference".format(torch.cuda.device_count()))
    #         model = nn.DataParallel(model)
    #     model.to(device)

    model.to(device)
    model.eval()

    # for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(
    #     val_loader
    # ):

    with torch.no_grad():
        img = img.to(device)
        if cfg.MODEL.SIE_CAMERA:
            camids = camids.to(device)
        else:
            camids = None
        if cfg.MODEL.SIE_VIEW:
            target_view = target_view.to(device)
        else:
            target_view = None
        feat = model(img, cam_label=camids, view_label=target_view)  # Extract features
    return feat


def set_reid_model():
    # parser = argparse.ArgumentParser(description="ReID Baseline Training")
    # parser.add_argument(
    #     "--config_file",
    #     default="/homes/hnakayama/tracking_practice/gta/clip_reid/configs/person/vit_clipreid.yml",
    #     help="path to config file",
    #     type=str,
    # )
    # parser.add_argument(
    #     "opts",
    #     help="Modify config options using the command-line",
    #     default=None,
    #     nargs=argparse.REMAINDER,
    # )

    # args = parser.parse_args()

    # if args.config_file != "":
    #     cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    # cfg.freeze()

    cfg.merge_from_file(
        "/homes/hnakayama/tracking_practice/gta/clip_reid/configs/person/vit_clipreid.yml"
    )
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID

    # (
    #     train_loader,
    #     train_loader_normal,
    #     val_loader,
    #     num_query,
    #     num_classes,
    #     camera_num,
    #     view_num,
    # ) = make_dataloader(cfg)

    # _, num_classes, camera_num, view_num = get_info(cfg)
    num_query, num_classes, camera_num, view_num = 2228, 702, 8, 1

    model = make_model(
        cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num
    )
    model.load_param(
        "/homes/hnakayama/tracking_practice/gta/Duke_clipreid_ViT-B-16_60.pth"
    )

    return model


def get_feat(model, img):
    trans_img = set_img(cfg, img)
    feat = do_inference(cfg, model, trans_img)

    return feat
