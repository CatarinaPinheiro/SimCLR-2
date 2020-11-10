import os
import argparse
import torch
import torchvision
import torch.utils.data.dataloader
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

from utils import yaml_config_hook, metrics_seg

from modules import SimCLR, get_resnet, fcn_segmentation
from modules.transformations import TransformsSimCLR
from datasets import transform_data, voc

writer = SummaryWriter()

config_for_train = {"trainer": {
    "epochs": 80,
    "save_dir": "saved/",
    "save_period": 10,

    "monitor": "max Mean_IoU",
    "early_stop": 10,

    "tensorboard": True,
    "log_dir": "saved/runs",
    "log_per_iter": 20,

    "val": True,
    "val_per_epochs": 5
}}

'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask, palette):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def train(args, loader, model, criterion, optimizer, nclass, config):
    loss_epoch = 0
    batch_time, data_time, total_loss, total_inter, total_union, total_correct, \
    total_label = metrics_seg._reset_metrics()
    wrt_mode, wrt_step = 'train_', 0
    log_step = config["trainer"].get('log_per_iter', int(np.sqrt(train_loader.batch_size)))
    if config['trainer']['log_per_iter']: log_step = int(log_step / train_loader.batch_size) + 1
    val_visual = []

    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        print('x.shape: ', x.cpu().numpy().shape)
        y = y.to(args.device).long()
        y = y.squeeze(1)
        print('y.shape: ', y.cpu().numpy().shape)

        # plt.imshow(np.swapaxes(x.cpu().numpy(), 0, 2))
        # plt.imshow(np.swapaxes(y.cpu().numpy(), 0, 2))
        # plt.show()
        output = model(x)
        print(output.shape)
        #predicted = torch.argmax(output, dim=1)
        # print("output.shape", output.cpu().detach().numpy().shape)
        # print("output.argmax.shape", torch.argmax(output, dim=1).cpu().detach().numpy().shape)

        loss = criterion(output, y)

        if len(val_visual) < 15:
            target_np = y.data.cpu().numpy()
            output_np = torch.argmax(output, dim=1).cpu().detach().numpy()
            val_visual.append([x[0].data.cpu(), target_np[0], output_np[0]])
            pass
        val_img = []

        restore_transform = transforms.Compose([
            transforms.ToPILImage()])
        viz_transform = transforms.Compose([
            # transforms.Resize((400, 400)),
            transforms.ToTensor()])

        for d, t, o in val_visual:
            d = restore_transform(d)
            t, o = colorize_mask(t, palette), colorize_mask(o, palette)
            d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
            val_img.extend([d, t, o])
            [d, t, o] = [viz_transform(x) for x in [d, t, o]]

            # plt.imshow(np.swapaxes(d.numpy(), 0, 2))
            # plt.imshow(np.swapaxes(t.numpy(), 0, 2))
            # plt.imshow(np.swapaxes(o.numpy(), 0, 2))
            # plt.show()

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()

        # Tensorboard
        if step % log_step == 0:
            wrt_step = (epoch - 1) * len(train_loader) + step
            writer.add_scalar(f'{wrt_mode}/loss', loss.item(), wrt_step)

        # Eval metrics

        seg_metrics = metrics_seg.eval_metrics(output, y, nclass)
        correct, labeled, inter, union = seg_metrics
        total_correct += correct
        total_label += labeled
        total_inter += inter
        total_union += union
        pixAcc_epoch, mIoU_epoch, _ = metrics_seg._get_seg_metrics(total_correct, total_label,
                                                                   total_inter, total_union, nclass).values()

    # Metrics to tensorboard
    seg_metrics = metrics_seg._get_seg_metrics(total_correct, total_label,
                                               total_inter, total_union, nclass)
    for k, v in list(seg_metrics.items())[:-1]:
        writer.add_scalar(f'{wrt_mode}/{k}', v, wrt_step)
    for i, opt_group in enumerate(optimizer.param_groups):
        writer.add_scalar(f'{wrt_mode}/Learning_rate_{i}', opt_group['lr'], wrt_step)
        # self.writer.add_scalar(f'{self.wrt_mode}/Momentum_{k}', opt_group['momentum'], self.wrt_step)

    # Metrics to return
    pixAcc, mIoU, _ = metrics_seg._get_seg_metrics(total_correct, total_label,
                                                   total_inter, total_union, nclass).values()
    return loss_epoch, mIoU, pixAcc


def test(args, loader, model, criterion, optimizer, nclass):
    loss_epoch = 0
    wrt_mode, wrt_step = 'val', 0
    batch_time, data_time, total_loss, total_inter, total_union, total_correct, total_label = metrics_seg._reset_metrics()
    val_visual = []
    model.eval()

    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device).long().squeeze(1)
        target = y
        data = x

        output = model(x)
        loss = criterion(output, y)

        loss_epoch += loss.item()

        # Eval metrics
        seg_metrics = metrics_seg.eval_metrics(output, y, nclass)
        correct, labeled, inter, union = seg_metrics
        total_correct += correct
        total_label += labeled
        total_inter += inter
        total_union += union

        # LIST OF IMAGE TO VIZ (15 images)
        if len(val_visual) < 15:
            target_np = target.data.cpu().numpy()
            output_np = output.data.max(1)[1].cpu().numpy()
            val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

        #
        pixAcc_epoch, mIoU_epoch, _ = metrics_seg._get_seg_metrics(total_correct, total_label,
                                                                   total_inter, total_union, nclass).values()

    val_img = []

    restore_transform = transforms.Compose([
        transforms.ToPILImage()])
    viz_transform = transforms.Compose([
        # transforms.Resize((400, 400)),
        transforms.ToTensor()])

    for d, t, o in val_visual:
        d = restore_transform(d)
        t, o = colorize_mask(t, palette), colorize_mask(o, palette)
        d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
        [d, t, o] = [viz_transform(x) for x in [d, t, o]]
        val_img.extend([d, t, o])
    val_img = torch.stack(val_img, 0)
    val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
    writer.add_image(f'{wrt_mode}/inputs_targets_predictions', val_img, wrt_step)

    # METRICS TO TENSORBOARD
    wrt_step = epoch * len(loader)
    writer.add_scalar(f'{wrt_mode}/loss', total_loss.average, wrt_step)
    seg_metrics = metrics_seg._get_seg_metrics(total_correct, total_label,
                                               total_inter, total_union, nclass)
    for k, v in list(seg_metrics.items())[:-1]:
        writer.add_scalar(f'{wrt_mode}/{k}', v, wrt_step)

    # Metrics to return
    val_pixAcc, val_mIoU, _ = metrics_seg._get_seg_metrics(total_correct, total_label,
                                                           total_inter, total_union, nclass).values()

    return loss_epoch, val_mIoU, val_pixAcc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    #n_classes = 21
    test_transform = torchvision.transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]
    )

    target_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.CenterCrop(args.image_size),
            torchvision.transforms.ToTensor(),
            # transform_data.ToLabel(),
            # transform_data.Relabel(255, n_classes),
        ]
    )

    # if args.dataset == "VOC":
    # train_dataset = voc.VOC12(root=args.dataset_dir, image_set="train", input_transform=test_transform,
    # target_transform=target_transform)
    # test_dataset = voc.VOC12(root=args.dataset_dir, image_set="val", input_transform=test_transform,
    #  target_transform=target_transform)
    if args.dataset == "VOC":
        train_dataset = torchvision.datasets.VOCSegmentation(
            args.dataset_dir,
            image_set="train",
            transform=test_transform,
            target_transform=target_transform,
        )
        test_dataset = torchvision.datasets.VOCSegmentation(
            args.dataset_dir,
            image_set="val",
            transform=test_transform,
            target_transform=target_transform,
        )
    elif args.dataset == "cityscapes":
        train_dataset = torchvision.datasets.Cityscapes(
            args.dataset_dir,
            split="train",
            target_type='semantic',
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.Cityscapes(
            args.dataset_dir,
            split="test",
            target_type='semantic',
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )

    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=True,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            train=False,
            download=True,
            transform=TransformsSimCLR(size=args.image_size).test_transform,
        )
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.logistic_batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
    )

    encoder = get_resnet(args.resnet, pretrained=False)  # encoder network
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # load pre-trained model from checkpoint
    n_classes = 21
    simclr_model = SimCLR(args, encoder, n_features)
    model_fp = os.path.join(
        args.model_path, "checkpoint_{}.tar".format(args.epoch_num)
    )  # load from checkpoint
    simclr_model.load_state_dict(torch.load(model_fp, map_location=args.device.type))  # load pretrained model
    simclr_model = simclr_model.to(args.device)  # run cuda
    simclr_model.eval()

    model_weights = os.path.join(args.model_path, "weights_100")  # weights from pretext task
    pretrained_model = fcn_segmentation.FeatureResNet()
    loaded = torch.load(model_weights)
    pretrained_model.load_state_dict(loaded)

    # FCN Segmentation
    model = fcn_segmentation.SegResNet(n_classes, pretrained_net=pretrained_model)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # SGD 1e-4, .9, 2e-5
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    for epoch in range(args.logistic_epochs):
        loss_epoch, miou_epoch, accuracy_pixel_epoch = train(
            args, train_loader, model, criterion, optimizer, n_classes, config=config_for_train
        )
        print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(train_loader)}\t"
            f" Accuracy: {accuracy_pixel_epoch / len(train_loader)} \t mIoU: {miou_epoch / len(train_loader)}\t"
        )

    # final testing
    loss_epoch, miou_epoch, accuracy_pixel_epoch = test(
        args, test_loader, model, criterion, optimizer, n_classes
    )
    print(
        f"[FINAL]\t Loss: {loss_epoch / len(test_loader)}\t Accuracy: {accuracy_pixel_epoch} \t mIoU: {miou_epoch}"
    )
