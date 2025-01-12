import torch
import argparse

from yolact import Yolact
from utils.functions import SavePath
from utils.augmentations import FastBaseTransform
from data import cfg, set_cfg

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='convert YOLACT .pth to .onnx')
    
    parser.add_argument('--trained_model',
                        type = str,
                        help = 'Trained state_dict file path to convert.'
                        )

    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_args()

    model_path = SavePath.from_str(args.trained_model)

    # TODO (from the original eval.py): Bad practice? Probably want to do a name lookup instead.
    args.config = model_path.model_name + '_config'
    print('Config not specified. Parsed %s from the file name.\n' % args.config)
    set_cfg(args.config)

    with torch.no_grad():
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()

        # If I not feedforward a signal into the network the exported onnx give very obscure results and I have no idea why.
        pred_outs = net(FastBaseTransform()(torch.randn(1, 640, 640,3)))

        torch.onnx.export(net, torch.autograd.Variable(torch.randn(1, 3, 640, 640)), args.trained_model.replace('.pth', '.onnx'),opset_version=11,do_constant_folding=True,keep_initializers_as_inputs=True,verbose=True)


# USAGE: python export_onnx.py --trained_model=weights/yolact_base_71_7119_interrupt.pth 
