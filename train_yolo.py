from __future__ import division
from __future__ import print_function
import os
import sys
import cv2
import random
import pickle
import argparse
import time

import torch
import torch.utils.data as data
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler

from dataloader import COCO_dataloader, detection_collate
from models import *
from utils.utils import *
from utils.datasets import *
from test import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="outputs/", help="path to save the model")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model config file")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--data_root", type=str, default="/data/Docker_Data/COCO/", help="path to data folder")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--num_iters", type=int, default=20, help="number of iterations")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    data_config = parse_data_config(opt.data_config)
    train_path =
    

    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    optimizer = optim.Adam(model.parameters())

    dataloader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            num_workers=opt.n_cpu,
                            shuffle=True,
                            collate_fn=detection_collate,
                            pin_memory=True)

    metrics = [
                "grid_size",
                "loss",
                "x",
                "y",
                "w",
                "h",
                "conf",
                "cls",
                "cls_acc",
                "recall50",
                "recall75",
                "precision",
                "conf_obj",
                "conf_noobj",
                ]

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()

        for batch_i, (imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.cuda(), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ------------------
            #    Log progress
            # ------------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" %  (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]


            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)




