import torch
import argparse
import typing as t
import numpy as np
import torchio as tio
from time import time
from tqdm import tqdm
from pathlib import Path
from shutil import rmtree
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler

from supermri.metrics import metrics
from supermri.utils import utils, data
from supermri.critic.critic import Critic
from supermri.models.registry import get_model
from supermri.utils.tensorboard import Summary


def train_step(args,
               inputs,
               targets,
               model,
               loss_function,
               optimizer,
               scaler,
               critic=None) -> dict:
  result = {}

  model.train()
  optimizer.zero_grad()

  with autocast(enabled=args.mixed_precision):
    logits = model(inputs)
    outputs = F.sigmoid(logits) if args.output_logits else logits
    loss = loss_function(logits, targets)

  result.update({'loss/sr_loss': loss.detach().clone()})

  if critic is not None and args.critic_loss > 0:
    critic_score = critic.predict(outputs)
    critic_loss = args.critic_loss * (1.0 - critic_score)
    result.update({'loss/critic_loss': critic_loss.detach().clone()})
    loss += critic_loss

  scaler.scale(loss).backward()
  scaler.step(optimizer)
  scaler.update()

  outputs, targets = outputs.detach(), targets.detach()

  result.update({
      'loss/total_loss': loss.detach().clone(),
      'metrics/NMSE': metrics.nmse(outputs, targets)
  })

  # train critic
  if critic is not None:
    critic_result = critic.train(real=targets, fake=outputs)
    result.update(critic_result)

  return result


def train(args, data_loader, model, critic, optimizer, loss_function, scaler,
          summary, epoch: int) -> dict:
  results = {}

  model.train()
  for batch in tqdm(data_loader, desc='Train', disable=args.verbose == 0):
    inputs, targets = data.prepare_batch(batch,
                                         dim=args.slice_dim,
                                         device=args.device)
    result = train_step(args,
                        inputs=inputs,
                        targets=targets,
                        model=model,
                        loss_function=loss_function,
                        optimizer=optimizer,
                        scaler=scaler,
                        critic=critic)
    utils.update_dict(results, result)

  for key, value in results.items():
    results[key] = torch.stack(value).mean()
    summary.scalar(key, results[key], step=epoch, mode=0)

  return results


def validation_step(args, inputs, targets, model, critic,
                    loss_function) -> dict:
  result = {}
  model.eval()

  with torch.no_grad():
    with autocast(enabled=args.mixed_precision):
      logits = model(inputs)
      outputs = F.sigmoid(logits) if args.output_logits else logits
      loss = loss_function(logits, targets)

    result.update({'loss/sr_loss': loss.clone()})

    if critic is not None and args.critic_loss > 0:
      critic_score = critic.predict(outputs)
      critic_loss = args.critic_loss * (1.0 - critic_score)
      result.update({'loss/critic_loss': critic_loss.clone()})
      loss += critic_loss

  if args.mixed_precision:
    outputs, targets = outputs.float(), targets.float()

  result.update({
      'loss/total_loss': loss.clone(),
      'metrics/MAE': metrics.mae(outputs, targets),
      'metrics/NMSE': metrics.nmse(outputs, targets),
      'metrics/PSNR': metrics.psnr(outputs, targets),
      'metrics/SSIM': metrics.ssim(outputs, targets)
  })

  # validate critic
  if critic is not None:
    critic_result = critic.validate(real=targets, fake=outputs)
    result.update(critic_result)

  return result


def validate(args, data_loader, model, critic, loss_function, summary,
             epoch: int) -> dict:
  results = {}

  for batch in tqdm(data_loader, desc='Validation', disable=args.verbose == 0):
    inputs, targets = data.prepare_batch(batch,
                                         dim=args.slice_dim,
                                         device=args.device)
    result = validation_step(args,
                             inputs=inputs,
                             targets=targets,
                             model=model,
                             critic=critic,
                             loss_function=loss_function)
    utils.update_dict(results, result)

  for key, value in results.items():
    results[key] = torch.stack(value).mean()
    summary.scalar(key, results[key], step=epoch, mode=1)

  return results


def test(args, model, loss_function, summary, epoch: int = 0):
  '''
  Test args.test_filenames and save metrics to args.output_dir/test_results.csv
  '''
  if args.verbose:
    print(f'\nInference {len(args.test_filenames)} scans from test set')

  results = {}
  model.eval()

  for filename in args.test_filenames:
    subject = data.load_subject(lr_filename=filename,
                                sequence=None,
                                require_hr=True)
    sampler = tio.GridSampler(subject=subject, patch_size=args.patch_shape)
    data_loader = DataLoader(sampler, batch_size=args.batch_size)
    aggregator = tio.GridAggregator(sampler, overlap_mode='average')

    sr_losses = []

    for batch in tqdm(data_loader, desc=subject.name):
      inputs, targets = data.prepare_batch(batch,
                                           dim=args.slice_dim,
                                           device=args.device)
      with torch.no_grad():
        if args.combine_sequence:
          outputs = model(inputs)
          if args.output_logits:
            outputs = F.sigmoid(outputs)
          sr_loss = loss_function(outputs, targets)
        else:
          # inference each channel separately and combine them
          outputs = torch.zeros_like(targets)
          channels_loss = []
          for channel in range(targets.shape[1]):
            channel_input = torch.unsqueeze(inputs[:, channel], dim=1)
            channel_target = torch.unsqueeze(targets[:, channel], dim=1)
            channel_output = model(channel_input)
            if args.output_logits:
              channel_output = F.sigmoid(channel_output)
            outputs[:, channel] = channel_output[:, 0]
            channels_loss.append(loss_function(channel_output, channel_target))
          sr_loss = torch.stack(channels_loss).mean()

      sr_losses.append(sr_loss)

      outputs = torch.unsqueeze(outputs, dim=args.slice_dim)
      aggregator.add_batch(outputs, batch[tio.LOCATION])

    output_tensor = aggregator.get_output_tensor()
    input_tensor = subject['lr'][tio.DATA]
    target_tensor = subject['hr'][tio.DATA]

    utils.update_dict(
        results, {
            'loss/sr_loss': torch.stack(sr_losses).mean(),
            'metrics/MAE': metrics.mae(output_tensor, target_tensor),
            'metrics/NMSE': metrics.nmse(output_tensor, target_tensor),
            'metrics/PSNR': metrics.psnr(output_tensor, target_tensor),
            'metrics/SSIM': metrics.ssim(output_tensor, target_tensor)
        })

    summary.plot_stitched(f'stitched/{subject.name}',
                          samples={
                              'inputs': input_tensor,
                              'targets': target_tensor,
                              'outputs': output_tensor
                          },
                          dim=args.slice_dim - 1,
                          step=epoch,
                          mode=2)

  for key, value in results.items():
    results[key] = torch.stack(value).mean()
    summary.scalar(key, results[key], step=epoch, mode=2)

  if args.verbose:
    print(f'Loss: {results["loss/sr_loss"]:.04f}\t'
          f'MAE: {results["metrics/MAE"]:.04f}\t'
          f'PSNR: {results["metrics/PSNR"]:.02f}\t'
          f'SSIM: {results["metrics/SSIM"]:.04f}\n')

  utils.save_csv(filename=args.output_dir / 'test_results.csv', data=results)


def main(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  args.output_dir = Path(args.output_dir)
  # delete args.output_dir if the flag is set and the directory exists
  if args.clear_output_dir and args.output_dir.exists():
    rmtree(args.output_dir)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  args.checkpoint_dir = args.output_dir / 'checkpoints'
  args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

  args.cuda = not args.no_cuda and torch.cuda.is_available()
  args.device = torch.device("cuda" if args.cuda else "cpu")

  train_loader, val_loader = data.get_loaders(args)

  # select random slice/patch for plotting
  samples = data.random_samples(args, val_loader)

  summary = Summary(args)

  # gradient scaling for mixed precision training
  scaler = GradScaler(enabled=args.mixed_precision)

  model = get_model(args, summary)
  optimizer = optim.Adam(model.parameters(), lr=args.lr)
  scheduler = StepLR(optimizer,
                     step_size=args.lr_step_size,
                     gamma=args.lr_gamma)
  loss_function = utils.get_loss_function(name=args.loss)

  critic = None if args.critic is None else Critic(args, summary=summary)

  utils.save_args(args)

  epoch = 0
  if args.checkpoint_dir.exists():
    model, epoch = utils.load_checkpoint(args, model)

  utils.plots(args,
              model=model,
              critic=critic,
              samples=samples,
              summary=summary,
              epoch=epoch)

  while (epoch := epoch + 1) < args.epochs + 1:
    if args.verbose:
      print(f'Epoch {epoch:03d}/{args.epochs:03d}')

    start = time()
    train_results = train(args,
                          data_loader=train_loader,
                          model=model,
                          critic=critic,
                          optimizer=optimizer,
                          loss_function=loss_function,
                          scaler=scaler,
                          summary=summary,
                          epoch=epoch)
    val_results = validate(args,
                           data_loader=val_loader,
                           model=model,
                           critic=critic,
                           loss_function=loss_function,
                           summary=summary,
                           epoch=epoch)
    end = time()

    if epoch % 10 == 0 or epoch + 1 == args.epochs:
      utils.plots(args,
                  model=model,
                  critic=critic,
                  samples=samples,
                  summary=summary,
                  epoch=epoch)
      utils.save_model(args, model=model, epoch=epoch)

    summary.scalar('model/elapse', end - start, step=epoch, mode=0)
    summary.scalar('model/learning_rate',
                   scheduler.get_last_lr()[0],
                   step=epoch,
                   mode=0)
    summary.scalar('model/gradient_scale',
                   scaler.get_scale(),
                   step=epoch,
                   mode=0)

    scheduler.step()

    if args.verbose:
      print(f'Train\t\tLoss: {train_results["loss/sr_loss"]:.04f}\n'
            f'Validation\tLoss: {val_results["loss/sr_loss"]:.04f}\t'
            f'MAE: {val_results["metrics/MAE"]:.04f}\t'
            f'PSNR: {val_results["metrics/PSNR"]:.02f}\t'
            f'SSIM: {val_results["metrics/SSIM"]:.04f}\n'
            f'Elapse: {end - start:.2f}s\n')

  utils.save_model(args, model=model, epoch=epoch)

  test(args,
       model=model,
       loss_function=loss_function,
       summary=summary,
       epoch=epoch)

  summary.close()

  print(f'tensorboard summary and model checkpoint saved to {args.output_dir}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir',
                      type=str,
                      required=True,
                      help='path to dataset')
  parser.add_argument('--output_dir',
                      type=str,
                      required=True,
                      help='directory to write TensorBoard summary.')

  # SR model settings
  parser.add_argument('--model',
                      type=str,
                      default='agunet',
                      help='model to use')
  parser.add_argument('--num_filters',
                      type=int,
                      default=64,
                      help='number of filters or hidden units (default: 64)')
  parser.add_argument('--normalization',
                      type=str,
                      default='instancenorm',
                      help='normalization layer (default: instancenorm)')
  parser.add_argument('--activation',
                      type=str,
                      default='leakyrelu',
                      help='activation layer (default: leakyrelu)')
  parser.add_argument('--dropout',
                      type=float,
                      default=0.0,
                      help='dropout rate (default 0.0)')

  # critic model settings
  parser.add_argument('--critic',
                      type=str,
                      default=None,
                      help='adversarial loss to use.')
  parser.add_argument('--critic_num_filters',
                      type=int,
                      default=None,
                      help='number of filters or hidden units in critic model')
  parser.add_argument('--critic_num_blocks',
                      type=int,
                      default=1,
                      help='number of blocks in DCGAN critic model')
  parser.add_argument('--critic_dropout',
                      type=float,
                      default=0.2,
                      help='critic model dropout rate')
  parser.add_argument('--critic_lr',
                      type=float,
                      default=0.0002,
                      help='critic model learning rate')
  parser.add_argument('--critic_steps',
                      type=int,
                      default=1,
                      help='number of update steps for critic per global step')
  parser.add_argument('--critic_loss',
                      type=float,
                      default=0.0,
                      help='critic loss coefficient to the training objective')
  parser.add_argument('--label_smoothing', action='store_true')

  # learning rate scheduler settings
  parser.add_argument('--lr',
                      type=float,
                      default=0.001,
                      metavar='LR',
                      help='learning rate (default: 0.001)')
  parser.add_argument('--lr_step_size',
                      type=int,
                      default=20,
                      help='learning rate decay step size (default: 10)')
  parser.add_argument('--lr_gamma',
                      type=float,
                      default=0.5,
                      help='learning rate step gamma (default: 0.5)')

  # input settings
  parser.add_argument('--patch_size',
                      type=int,
                      default=None,
                      help='patch size, None to train on the entire scan.')
  parser.add_argument('--n_patches',
                      type=int,
                      default=None,
                      help='number of patches to generate per sample')
  parser.add_argument('--combine_sequence',
                      action='store_true',
                      help='combine FLAIR, T1 and T2 as a single input')

  parser.add_argument('--batch_size',
                      type=int,
                      default=32,
                      metavar='N',
                      help='batch size for training (default: 32)')
  parser.add_argument('--epochs',
                      type=int,
                      default=100,
                      help='number of epochs (default: 100)')
  parser.add_argument('--loss',
                      type=str,
                      default='bce',
                      help='loss function to use (default: bce)')
  parser.add_argument('--no_cuda',
                      action='store_true',
                      default=False,
                      help='disables CUDA training')
  parser.add_argument('--seed',
                      type=int,
                      default=42,
                      help='random seed (default: 42)')
  parser.add_argument('--mixed_precision',
                      action='store_true',
                      help='use mixed precision compute')
  parser.add_argument('--num_workers',
                      type=int,
                      default=4,
                      help='number of workers for data loader')
  parser.add_argument('--extension',
                      type=str,
                      default='npy',
                      choices=['npy', 'mat'],
                      help='MRI scan file extension')
  parser.add_argument('--save_plots',
                      action='store_true',
                      help='save TensorBoard figures and images to disk.')
  parser.add_argument('--dpi',
                      type=int,
                      default=120,
                      help='DPI of matplotlib figures')
  parser.add_argument('--clear_output_dir',
                      action='store_true',
                      help='overwrite existing output directory')
  parser.add_argument('--verbose', choices=[0, 1, 2], default=1, type=int)
  params = parser.parse_args()

  main(params)
