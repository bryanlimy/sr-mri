import os
import re
import csv
import json
import copy
import torch
import warnings
import subprocess
import numpy as np
import typing as t
from math import ceil
from pathlib import Path
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt

from supermri.metrics import metrics
from supermri.utils.gradcam import GradCAM
from supermri.utils.tensorboard import Summary
from supermri.utils.attention_gate import AGHook


def get_loss_function(name: str):
  """
  Args:
    name: name of loss function
  Returns:
    loss function
  """
  name = name.lower()
  if name in ['mse', 'meansquarederror', 'l2']:
    return F.mse_loss
  elif name in ['mae', 'meanabsoluteerror', 'l1']:
    return F.l1_loss
  elif name in ['bce', 'binarycrossentropy']:
    return F.binary_cross_entropy_with_logits
  elif name in ['ssim']:
    return lambda x, y: 1 - metrics.ssim(x, y)
  raise ValueError(f'Unknown loss function name {name}')


def get_current_git_hash():
  """ return the current Git hash """
  try:
    return subprocess.check_output(['git', 'describe',
                                    '--always']).strip().decode()
  except Exception:
    warnings.warn('Unable to get git hash.')


def save_json(filename: Path, data: t.Dict):
  """ Save dictionary data to filename as a json file """
  assert type(data) == dict
  for key, value in data.items():
    if isinstance(value, np.ndarray):
      data[key] = value.tolist()
    elif isinstance(value, np.float32):
      data[key] = float(value)
    elif isinstance(value, Path) or isinstance(value, torch.device):
      data[key] = str(value)
  with open(filename, 'w') as file:
    json.dump(data, file)


def load_json(filename: Path) -> t.Dict:
  """ Load json file as a dictionary"""
  with open(filename, 'r') as file:
    data = json.load(file)
  return data


def update_json(filename: Path, data: t.Dict):
  """ Update json file with items in data """
  content = {}
  if os.path.exists(filename):
    content = load_json(filename)
  for key, value in data.items():
    content[key] = value
  save_json(filename, content)


def save_args(args):
  """ Save input arguments as a json file in args.output_dir"""
  args.git_hash = get_current_git_hash()
  save_json(args.output_dir / 'args.json', copy.deepcopy(args.__dict__))


def load_args(args, filename=None):
  """ Load input arguments from filename """
  if filename is None:
    filename = args.output_dir / 'args.json'
  content = load_json(filename)
  for key, value in content.items():
    if (not hasattr(args, key)) or (getattr(args, key) == None):
      setattr(args, key, value)


def save_csv(filename, data: t.Dict[str, t.Union[torch.Tensor, np.ndarray]]):
  content = {
      key: data[key].item() if torch.is_tensor(data[key]) else float(data[key])
      for key in sorted(data.keys())
  }
  with open(filename, 'w') as file:
    writer = csv.DictWriter(file, list(content.keys()))
    writer.writeheader()
    writer.writerow(content)


def save_model(args, model, epoch: int):
  """ save model state_dict to args.checkpoint_dir """
  filename = args.checkpoint_dir / f'model_epoch{epoch:03d}.pt'
  torch.save(model.state_dict(), filename)
  print(f'\nSaved checkpoint to {filename}\n')


def load_checkpoint(args, model):
  """ Loads model weights from checkpoints folder. """
  epoch = 0
  if type(args.checkpoint_dir) is not Path:
    args.checkpoint_dir = Path(args.checkpoint_dir)
  checkpoints = sorted(list(args.checkpoint_dir.glob('*.pt')))
  if checkpoints:
    checkpoint = checkpoints[-1]
    match = re.match(r'model_epoch(\d{3}).pt', checkpoint.name)
    epoch = int(match.groups()[0])
    model.load_state_dict(torch.load(checkpoint, map_location=args.device))
    if args.verbose:
      print(f'\nLoaded checkpoint from {checkpoint} at epoch {epoch:03d}\n')
  return model, epoch


def to_numpy(x: t.Union[np.ndarray, torch.tensor]) -> np.ndarray:
  """ Convert torch.tensor to CPU numpy array"""
  return (x.cpu().numpy()).astype(np.float32) if torch.is_tensor(x) else x


def update_dict(source: dict, target: dict, replace: bool = False):
  """ replace or append items in target to source """
  for key, value in target.items():
    if replace:
      source[key] = value
    else:
      if key not in source:
        source[key] = []
      source[key].append(value)


def save_array_as_pdf(filename: t.Union[Path, str],
                      array: t.Union[np.ndarray, torch.Tensor],
                      dpi: int = 120):
  """
  Save 2D array as PDF image

  Args:
    filename: name of the output file
    img: 2D image
    dpi: dpi of the output image
  """
  assert len(array.shape) == 2
  if torch.is_tensor(array):
    array = array.numpy()
  figure, ax = plt.subplots(1, figsize=(8, 8), squeeze=False, dpi=dpi)
  axes = ax.ravel()
  axes[0].imshow(array, cmap='gray', interpolation='none')
  axes[0].axis('off')
  plt.tight_layout()
  figure.savefig(str(filename),
                 dpi=dpi,
                 format="pdf",
                 bbox_inches='tight',
                 pad_inches=0,
                 transparent=True)
  plt.close(figure)


def get_padding(shape1, shape2):
  """ Return the padding needed to convert shape2 to shape1 """
  assert len(shape1) == len(shape2)
  h_diff, w_diff = shape1[1] - shape2[1], shape1[2] - shape2[2]
  padding_left = w_diff // 2
  padding_right = w_diff - padding_left
  padding_top = h_diff // 2
  padding_bottom = h_diff - padding_top
  return (padding_left, padding_right, padding_top, padding_bottom)


def convert_square_shape(shape):
  """ Return new shape that are multiple 2 and square
  Args:
    shape: the existing shape
  Returns:
    new_shape: the new shape with square H,W and are multiple of 2
    padding: padding needed to convert shape to new shape
  """
  square_dim = 2 * ceil(max(shape[1:]) / 2)
  new_shape = (shape[0], square_dim, square_dim)
  return new_shape, get_padding(new_shape, shape)


def plot_attention_gate(args, model, samples, summary, epoch: int = 0):
  """ Plot AG-UNet attention masks with hook """
  if args.model != 'agunet':
    return

  inputs, _ = samples

  model_hook = AGHook(model, output_logits=args.output_logits)
  for i in range(len(inputs)):
    outputs = model_hook(torch.unsqueeze(inputs[i], dim=0))
    summary.plot_attention_masks(f'attention_gates/{i:02d}',
                                 inputs=inputs[i],
                                 outputs=outputs[0],
                                 gate_inputs=model_hook.gate_inputs,
                                 gate_masks=model_hook.gate_masks,
                                 step=epoch)

  del model_hook


def critic_gradcam(args,
                   model,
                   critic,
                   samples,
                   summary: Summary,
                   epoch: int = 0):
  if critic is None or critic.name != 'dcgan':
    return

  inputs, targets = samples

  # up-sample input scans
  outputs = torch.zeros_like(targets)
  for i in range(len(inputs)):
    with torch.no_grad():
      output = model(torch.unsqueeze(inputs[i], dim=0))
      if args.output_logits:
        output = F.sigmoid(output)
      outputs[i] = output[0]

  gradcam = GradCAM(args, model=critic.model)

  real_scores, real_cams = gradcam.forward(inputs=targets, is_real=True)
  fake_scores, fake_cams = gradcam.forward(inputs=outputs, is_real=False)

  summary.plot_gradcam(
      tag='GradCAM D(real)',
      images=targets,
      cams=real_cams,
      titles=[f'D(target): {score:.4f}' for score in real_scores.numpy()],
      step=epoch)

  summary.plot_gradcam(
      tag='GradCAM D(fake)',
      images=outputs,
      cams=fake_cams,
      titles=[f'D(generated): {score:.4f}' for score in fake_scores.numpy()],
      step=epoch)

  del gradcam


def plots(args, model, critic, samples, summary, epoch: int):
  ''' plot to tensorboard '''
  inputs, targets = samples
  with torch.no_grad():
    outputs = model(inputs)
    if args.output_logits:
      outputs = F.sigmoid(outputs)
  summary.plot_side_by_side('side_by_side',
                            samples={
                                'inputs': inputs,
                                'targets': targets,
                                'outputs': outputs
                            },
                            step=epoch,
                            mode=1)
  summary.plot_difference_maps('diff_maps',
                               samples={
                                   'inputs': inputs,
                                   'targets': targets,
                                   'outputs': outputs
                               },
                               step=epoch,
                               mode=1)

  plot_attention_gate(args, model, samples, summary, epoch=epoch)
  critic_gradcam(args, model, critic, samples, summary, epoch=epoch)
