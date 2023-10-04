import torch
import argparse
import os
import logging
import torch.nn as nn
import torch.nn.functional as F

import wandb
wandb.login()

import pdb

from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torch.optim import RMSprop, lr_scheduler
from tqdm import tqdm
from torchsummary import summary

from models.unet_model import UNet
from datasets.dataloading import CarvanaDataset, BasicDataset
from losses.dice_score import dice_loss
from evaluate_unet import evaluate


def train_unet(
        args,
        model,
        device,
        epochs:int  = 5,
        batch_size:int = 1,
        learning_rate:float = 1e-5,
        val_percent:float = 0.1,
        save_chkpt:bool = True,
        work_dir:str = './results',
        img_scale:float = 0.5,
        amp: bool = True,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0):
     
    # Prepare dataset
    dir_img = os.path.join(args.datapath, 'imgs')
    dir_mask = os.path.join(args.datapath, 'masks')
    dir_checkpoint = os.path.join(args.work_dir, 'checkpoints')
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # Split dataset into train and validation
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset,
                              [n_train, n_val],
                              generator=torch.Generator().manual_seed(0))

    # Create data loaders
    loaders_args = dict(batch_size=batch_size,
                        num_workers=8,
                        pin_memory=True)
    train_loader = DataLoader(train, shuffle=False, **loaders_args)
    val_loader = DataLoader(val, shuffle=False, drop_last = True, **loaders_args)

    # TODO: loggin with wandb or mlflow
    experiment = wandb.init(project='Pruebas_unet', entity='miguelag99', name=args.exp_name)
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_chkpt, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_chkpt}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # Optim, loss, lr scheduler...
    optimizer = RMSprop(model.parameters(),
                        lr=learning_rate,
                        weight_decay=weight_decay,
                        momentum=momentum,
                        foreach = True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                               'max',
                                               patience=5)
    criterion = nn.CrossEntropyLoss() if model.n_class > 1 else nn.BCEWithLogitsLoss()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0

    # Train loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']

                assert imgs.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(imgs)

                    if model.n_class == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_class).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(imgs.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(imgs[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if save_chkpt:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint + '/unet_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5, help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--datapath', '-d', type=str, default='/home/robesafe/Datasets/carvana', help='Path to the dataset folder')
    parser.add_argument('--work_dir', '-w', type=str, default='/home/robesafe/workspace', help='Path to the workspace folder')
    parser.add_argument('--exp_name','-exp_n', type=str, default='first_train', help='Name of the wandb experiment')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    if not os.path.exists(args.work_dir + '/checkpoints'):
        os.makedirs(args.work_dir + '/checkpoints')

    model = UNet(n_channels=3, n_class=2, bilinear=args.bilinear)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    summary(model,(3, 640, 640))
    train_unet(args,model,device,batch_size=args.batch_size,work_dir=args.work_dir,amp=args.amp)
