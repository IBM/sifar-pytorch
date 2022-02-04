import argparse
import os
import scipy.ndimage
import torch

parser = argparse.ArgumentParser(description='Convert from ViT for different input size')
parser.add_argument('--mode', default='', type=str, metavar='MODE',
                    help='moco or byol')
parser.add_argument('model_path', default='', type=str, metavar='MODEL',
                    help='The pretrained model')


def from_contrastive_model(model_path, mode=None):
    
    model = torch.load(model_path, map_location='cpu')
    state_dict_name = 'model'

    if mode == 'moco':
        keyword = 'encoder_q.'
    elif mode == 'byol':
        keyword = 'online_encoder.'
    else:
        raise ValueError(f"Unknown mode: {mode}")

    state_dict = model[state_dict_name]
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith(keyword):
            # remove prefix
            state_dict[k[len(keyword):]] = state_dict[k]
            # delete renamed or unused k
        del state_dict[k]
    
    for k in state_dict.keys():
        print(k)

    model = {state_dict_name: state_dict}
    return model


def main():
    args = parser.parse_args()

    model = from_contrastive_model(args.model_path, args.mode)
    name = os.path.join(os.path.dirname(args.model_path), os.path.basename(args.model_path).split(".")[0] + f'_remove_contrastive_wrapper' + '.pth.tar')
    print(f"Save model to {name}")
    torch.save(model, name, _use_new_zipfile_serialization=False)


if __name__ == "__main__":
    main()
