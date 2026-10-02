'''
Configs for training & testing
Written by Whalechen
'''

import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root', default='./data', type=str, help='Root directory path of data'
    )
    parser.add_argument(
        '--img_list',
        default='./data/train.txt',
        type=str,
        help='Path for image list file',
    )
    parser.add_argument(
        '--n_seg_classes', default=2, type=int, help="Number of segmentation classes"
    )
    parser.add_argument(
        '--learning_rate',  # set to 0.001 when finetune
        default=0.001,
        type=float,
        help='Initial learning rate (divided by 10 while training by lr scheduler)',
    )
    parser.add_argument('--num_workers', default=4, type=int, help='Number of jobs')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')
    parser.add_argument(
        '--phase', default='train', type=str, help='Phase of train or test'
    )
    parser.add_argument(
        '--save_intervals', default=10, type=int, help='Interation for saving model'
    )
    parser.add_argument(
        '--n_epochs', default=200, type=int, help='Number of total epochs to run'
    )
    parser.add_argument('--input_D', default=56, type=int, help='Input size of depth')
    parser.add_argument('--input_H', default=448, type=int, help='Input size of height')
    parser.add_argument('--input_W', default=448, type=int, help='Input size of width')
    parser.add_argument(
        '--resume_path', default='', type=str, help='Path for resume model.'
    )
    parser.add_argument(
        '--pretrain_path', default=None, type=str, help='Path for pretrained model.'
    )
    parser.add_argument(
        '--new_layer_names',
        # default=['upsample1', 'cmp_layer3', 'upsample2', 'cmp_layer2', 'upsample3', 'cmp_layer1', 'upsample4', 'cmp_conv1', 'conv_seg'],
        default=['conv_seg'],
        type=list,
        help='New layer except for backbone',
    )
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.'
    )
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--gpu_id', nargs='+', type=int, default=[0], help='Gpu id lists'
    )
    parser.add_argument(
        '--model',
        default='medicalnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | ',
    )
    parser.add_argument(
        '--medicalnet_depth',
        default=50,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)',
    )
    parser.add_argument(
        '--medicalnet_backbone',
        default='resnet',
        type=str,
        help='Backbone of the medicalNet if used',
    )
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)',
    )
    parser.add_argument(
        '--ci_test', action='store_true', help='If true, ci testing is used.'
    )

    # Resnext
    parser.add_argument(
        '--n_classes',
        default=400,
        type=int,
        help='Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)',
    )  # default 400
    parser.add_argument(
        '--sample_size', default=128, type=int, help='Height and width of inputs'
    )  # default: 112
    parser.add_argument(
        '--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality'
    )
    parser.add_argument(
        '--sample_duration', default=128, type=int, help='Temporal duration of inputs'
    )  # default: 16

    # Only for compatibility with the parent script
    parser.add_argument('--run_name')

    args = parser.parse_args()

    return args
