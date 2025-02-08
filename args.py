import argparse

def confirm(args):
    print('confirm args information:')
    for key, value in vars(args).items():
        print(f"\t{key} = {value}")
    print("*" * 30)


def parse_channels(value):
    if value.lower() == 'none':
        return None
    return value.split(',')

def parse_args():
    parser = argparse.ArgumentParser(description="EEG_embedding")

    parser.add_argument("--dataset", 
                        help="dataset to train and evluate",
                        type=str, 
                        default='dreamer', 
                        choices=['dreamer', 'stew', 'crowd', 'isruc', 'hinss', 'b2014', 'b2015', 'sleepedf', 'hmc', 'seedv', 'tuab', 'tuev'])
    parser.add_argument("--model_name",
                        help="name of the model to use",
                        type=str,
                        default='EEG_Transformer_CL_VIB_Network',
                        choices=['EEG_CNN_Network', 'EEG_Transformer_Network', 'EEG_Transformer_VIB_Network', 
                                 'EEG_Transformer_CL_VIB_Network', 'EEG_Transformer_CL_VIB_Network_wo_FTV', 
                                 'EEG_Transformer_CL_VIB_Network_wo_WTV', 'EEG_Transformer_CL_Network']
                        )
    parser.add_argument("--batch_size", 
                        help="batch size",
                        type=int, 
                        default=512)
    parser.add_argument("--num_workers", 
                        help="number of workers for DataLoader",
                        type=int, 
                        default=4)
    parser.add_argument('--epochs', 
                        help='total number of epochs',
                        type=int, 
                        default=50)
    parser.add_argument('--lr', 
                        help='learning rate',
                        type=float, 
                        default=1e-3)
    parser.add_argument('--device', 
                        help='GPU device',
                        type=str, 
                        default='cuda:0')
    
    parser.add_argument('--nhead',
                        help="number of head in TransformerEncoderLayer",
                        type=int,
                        default=8)
    parser.add_argument('--d_model',
                        help='number of dimension of the input to TransformerEncoderLayer',
                        type=int,
                        default=None)
    
    parser.add_argument('--latent_dim',
                        help="dimention of latent representation, or the value of K in VIB",
                        type=int,
                        default=256)
    parser.add_argument('--feat_dim',
                        help="dimention of feature representation for Classification loss",
                        type=int,
                        default=2048)
    parser.add_argument('--proj_dim',
                        help="dimention of projected representation for InfoNCE",
                        type=int,
                        default=512)
    parser.add_argument('--alpha',
                        help="alpha weight for information bottleneck: I(t,y) + alpha * [ I(t_, t_ft) + I(t_, t_wt) ]",
                        type=float,
                        default=1e-3)
    parser.add_argument('--temperature',
                        help="temperature for contrastive learning",
                        type=float,
                        default=0.07)
    parser.add_argument('--beta',
                        help="beta weight for information bottleneck: I(t,y) - beta * I(t,x)",
                        type=float,
                        default=1e-3)

    parser.add_argument('--chunk_second', 
                        help='chunk time for each sample in second',
                        type=int, 
                        default=None)
    parser.add_argument('--freq_rate', 
                        help='frequency rate',
                        type=int, 
                        default=None)
    parser.add_argument('--overlap', 
                        help='The number of overlapping data points between different chunks when dividing EEG chunks',
                        type=int, 
                        default=0)
    parser.add_argument('--resample',
                         help='whether resample (downsample or upsample) to the specific args.freq_rate or not (only useful in seedv dataset)',
                         type=bool,
                         default=False)
    parser.add_argument('--selected_channels', 
                        help='Comma-separated list of channels (e.g., "AF3,F7,F3,FC5,T7") or "None"',
                        type=parse_channels, 
                        default=None)

    args = parser.parse_args()

    

    # const definition
    num_class_dict = {
        "dreamer": 5,
        "stew": 9,
        "crowd": 2,
        "isruc": 5,
        "hinss": 3,
        "b2014": 4,
        "b2015": 2,
        "sleepedf": 5,
        "hmc": 5,
        "seedv": 5,
        "tuab": 2,
        "tuev": 6
    }
    args.num_class = num_class_dict[args.dataset]

    num_channel_dict = {
        "dreamer": 14,
        "stew": 14,
        "crowd": 14,
        "isruc": 6,
        "hinss": 61,
        "b2014": 22,
        "b2015": 13,
        "sleepedf": 1,  # 2
        "hmc": 4,
        "seedv": 62,
        "tuab": 16,
        "tuev": 16
    }
    args.num_channel = num_channel_dict[args.dataset]

    if args.chunk_second is None:
        if args.dataset in ['dreamer', 'stew', 'crowd', 'hinss']:
            args.chunk_second = 2
        elif args.dataset in ['seedv']:
            args.chunk_second = 1
        elif args.dataset in ['tuab']:
            args.chunk_second = 10
        elif args.dataset in ['tuev']:
            args.chunk_second = 5
        elif args.dataset in ['b2014']:
            args.chunk_second = 4
        elif args.dataset in ['b2015']:
            args.chunk_second = 5
        elif args.dataset in ['isruc', 'sleepedf', 'hmc']:
            args.chunk_second = 30
    
    if args.freq_rate is None:
        if args.dataset in ['dreamer', 'stew', 'crowd']:
            args.freq_rate = 128
        elif args.dataset in ['seedv']:
            args.freq_rate = 1000
        elif args.dataset in ['tuab', 'tuev']:
            args.freq_rate = 200
        elif args.dataset in ['isruc']:
            args.freq_rate = 100
        elif args.dataset in ['b2014', 'hinss']:
            args.freq_rate = 250
        elif args.dataset in ['b2015']:
            args.freq_rate = 512
        elif args.dataset in ['sleepedf', 'hmc']:
            args.freq_rate = 100

    if args.d_model is None:
        args.d_model = args.chunk_second * args.freq_rate

    return args

args = parse_args()
confirm(args)
