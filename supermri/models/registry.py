from supermri.models import torchsummary
_MODELS = dict()


def register(name):
  """ Note: update __init__.py so that register works """

  def add_to_dict(fn):
    global _MODELS
    _MODELS[name] = fn
    return fn

  return add_to_dict


def get_model(args, summary=None):
  ''' Initialize model '''
  assert args.model in _MODELS.keys(), f'model {args.model} not found.'

  # model should output logits with BCE loss
  args.output_logits = (args.loss in ['bce', 'binarycrossentropy'] and
                        args.model != 'identity')

  model = _MODELS[args.model](args)
  model.to(args.device)

  if summary is not None:
    # get model summary and write to args.output_dir
    summary_readout, trainable_parameters = torchsummary.summary(
        model, input_size=args.input_shape, device=args.device)
    with open(args.output_dir / 'model.txt', 'w') as file:
      file.write(summary_readout)
    summary.scalar('model/trainable_parameters', trainable_parameters)
    if args.verbose == 2:
      print(summary_readout)

  return model
