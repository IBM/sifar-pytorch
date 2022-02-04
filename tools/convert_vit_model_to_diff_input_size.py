import argparse
import os
import scipy.ndimage
import torch

parser = argparse.ArgumentParser(description='Convert from ViT for different input size')
parser.add_argument('--model', default='', type=str, metavar='MODEL',
                    help='The pretrained model')
parser.add_argument('--ori_input_size', '--ois', default=[224], type=int, nargs='+')
parser.add_argument('--new_input_size', '--nis', default=[384], type=int, nargs='+')
parser.add_argument('--ori_patch_size', '--ops', default=[16], type=int, nargs='+')
parser.add_argument('--new_patch_size', '--nps', default=[16], type=int, nargs='+')
parser.add_argument('--ema', action='store_true',)
parser.add_argument('--remove_fc', action='store_true',)

# need to convert the pos embedding.
# https://github.com/google-research/vision_transformer/blob/f952d612e1b55d1099b2fbf87fc04218d5c4fe18/vit_jax/checkpoint.py#L185


def _convert_one_set(pos_embed, n_p, o_p):
    pos_embed = pos_embed.reshape(1, o_p, o_p, -1).permute(0, 3, 1, 2)
    pos_embed = torch.nn.functional.interpolate(
            pos_embed, size=(n_p, n_p), mode='bicubic', align_corners=False)
    new_pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)

    """
    pos_embed = pos_embed.squeeze(0).reshape(o_p, o_p, -1).cpu().numpy()
    zoom = (n_p / o_p, n_p / o_p, 1)
    new_pos_embed = scipy.ndimage.zoom(pos_embed, zoom, order=1)
    new_pos_embed = torch.tensor(new_pos_embed).reshape((1, n_p * n_p, -1))
    """

    return new_pos_embed


def convert(model_path, ori_input_size, ori_patch_size, new_input_size, new_patch_size, use_ema=False, remove_fc=False):

    model = torch.load(model_path, map_location='cpu')
    state_dict_name = 'model_ema' if use_ema else 'model'
    model = {state_dict_name: model[state_dict_name]}
    if remove_fc:
        model[state_dict_name].pop('head.weight', None)
        model[state_dict_name].pop('head.bias', None)

    for idx, (nis, nps, ois, ops) in enumerate(zip(new_input_size, new_patch_size, ori_input_size, ori_patch_size)):
        new_num_patches = (nis // nps) * (nis // nps)
        ori_num_patches = (ois // ops) * (ois // ops)
        
        if new_num_patches == ori_num_patches:
            continue
        else:
            print(f"Resize the pos embedding: num_pos in checkpoint: {ori_num_patches}, num_pos in model: {new_num_patches}", flush=True)
            if f'pos_embed.{idx}' in model[state_dict_name]:
                embed_name = f'pos_embed.{idx}'
            else:
                embed_name = 'pos_embed'
            ori_pos_embed = model[state_dict_name][embed_name]
            start_pos = ori_pos_embed.shape[1] - ori_num_patches
            cls_token_embed = ori_pos_embed[:, :start_pos, :]
            n_p = (nis // nps)
            o_p = (ois // ops)
            new_pos_embed = _convert_one_set(ori_pos_embed[:, start_pos:, :], n_p, o_p)
            out = torch.cat((cls_token_embed, new_pos_embed), dim=1)
            model[state_dict_name][embed_name] = out
            
        """
        ori_pos_embed = ori_pos_embed[:, start_pos:, :]  # remove cls
        ori_pos_embed = ori_pos_embed.squeeze(0).reshape(o_p, o_p, -1).cpu().numpy()
        zoom = (n_p / o_p, n_p / o_p, 1)
        new_pos_embed = scipy.ndimage.zoom(ori_pos_embed, zoom, order=1)
        new_pos_embed = torch.tensor(new_pos_embed).reshape((1, n_p * n_p, -1))
        out = torch.cat((cls_token_embed, new_pos_embed), dim=1)
        print(f"{model[state_dict_name]['pos_embed'].shape}, {out.shape}")
        model[state_dict_name]['pos_embed'] = out
        """
    return model


def main():
    args = parser.parse_args()

    model = convert(args.model, args.ori_input_size, args.ori_patch_size, args.new_input_size, args.new_patch_size, args.ema, args.remove_fc)
    name = os.path.join(os.path.dirname(args.model), os.path.basename(args.model).split(".")[0] + f'_{max(args.new_input_size)}' + ("_no_fc" if args.remove_fc else "") +  '.pth.tar')
    print(f"Save model to {name}")
    torch.save(model, name, _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    main()

