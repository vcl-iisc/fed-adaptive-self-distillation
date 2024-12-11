from utils_general import *
from utils_methods import *
from process_args import *
from tiny_imagenet import tinyImageNetDataset

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

# Dataset initialization

########
# For 'CIFAR100' experiments
#     - Change the dataset argument from CIFAR10 to CIFAR100.
########
# For 'mnist' experiments
#     - Change the dataset argument from CIFAR10 to mnist.
########
# For 'emnist' experiments
#     - Download emnist dataset from (https://www.nist.gov/itl/products-and-services/emnist-dataset) as matlab format and unzip it in "Data/Raw/" folder.
#     - Change the dataset argument from CIFAR10 to emnist.
########
# For Shakespeare experiments
# First generate dataset using LEAF Framework and set storage_path to the data folder
# storage_path = 'LEAF/shakespeare/data/'
#     - In IID use

# name = 'shakepeare'
# data_obj = ShakespeareObjectCrop(storage_path, dataset_prefix)

#     - In non-IID use
# name = 'shakepeare_nonIID'
# data_obj = ShakespeareObjectCrop_noniid(storage_path, dataset_prefix)
#########


# Generate IID or Dirichlet distribution
# IID

def main():
    args = parse_args()
    
    try:
        validate_args(args)
        print('Starting the run...')
    except ValueError as e:
        print(f'Error: {e}')
    
    #### select some random seed for fair comparison ######
    rand_seed = args.seed
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)

    data_obj = DatasetObject(
        dataset=args.dataset_name, 
        n_client=args.n_client, 
        unbalanced_sgm=args.unbalanced_sgm,
        rule=args.rule, 
        rule_arg=args.rule_arg
    )

    # Common config
    alg_name = args.alg_name
    model_name = args.model_name
    com_amount = args.com_amount
    batch_size = args.batch_size
    lr_decay_per_round = args.lr_decay_per_round
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    save_period = args.save_period
    act_prob = args.act_prob
    epoch = args.epoch
    print_per = 20
    # FedDyn
    alpha_coef = args.alpha_coef
    mu = args.mu
    
    print("args.alg_name:", alg_name)

    model_func = lambda pretrained=False: client_model(model_name, pretrained)
    init_model = model_func()
    # init_model = torch.nn.DataParallel(init_model)

    if not os.path.exists('Output/%s/%s_init_mdl.pt' % (data_obj.name, model_name)):
        print("New directory!")
        os.mkdir('Output/%s/' % (data_obj.name))
        torch.save(init_model.state_dict(), 'Output/%s/%s_init_mdl.pt' % (data_obj.name, model_name))
    else:
        init_model.load_state_dict(torch.load('Output/%s/%s_init_mdl.pt' % (data_obj.name, model_name)), strict=False)

    # --- FedDisco based --- #
    num_classes = 100 # for cifar-100init_model.load_state_dict
    client_cls_counts_npy = np.array([])
    # data_obj.clnt_y = data_obj.clnt_y.astype(int)
    # y_train = data_obj.clnt_y
    
    #print("diff_value:",np.sum(y_train-y_train_tmp))
    #exit()
    # for clnt in range(data_obj.n_client):
    #     unq, unq_cnt = np.unique(y_train[clnt], return_counts=True)
    #     unq_cnt = unq_cnt.astype(int)
    #     tmp_npy = np.zeros(num_classes)
    #     #print("unq_cnt:",type(unq_cnt))
    #     for i in range(len(unq)):
    #         tmp_npy[int(unq[i])] = unq_cnt[i]
    #     client_cls_counts_npy = np.concatenate((client_cls_counts_npy, tmp_npy), axis=0)

    client_cls_counts_npy = np.reshape(client_cls_counts_npy, (-1, num_classes))

    global_dist = np.ones(client_cls_counts_npy.shape[1])/client_cls_counts_npy.shape[1]

    common_params = [
        data_obj,
        act_prob,
        learning_rate,
        batch_size,
        epoch,
        com_amount,
        print_per,
        weight_decay,
        model_func,
        init_model,
        save_period,
        lr_decay_per_round,
        args,
        rand_seed
    ]

    if args.alg_name == 'FedDyn':
        _ = train_FedDyn(*common_params, alpha_coef=alpha_coef)

    elif alg_name == 'SCAFFOLD':
        n_data_per_client = np.concatenate(data_obj.clnt_x, axis=0).shape[0] / args.n_client
        n_iter_per_epoch = np.ceil(n_data_per_client / batch_size)
        n_minibatch = (epoch * n_iter_per_epoch).astype(np.int64)
        scaffold_params = common_params
        scaffold_params.print_per = print_per * n_iter_per_epoch

        _ = train_SCAFFOLD(*scaffold_params, n_minibatch=n_minibatch)

    elif alg_name == 'FedAvg':
        _ = train_FedAvg(*common_params, client_cls_counts=client_cls_counts_npy, global_dist=global_dist)

    elif alg_name == 'FedProx':
        _ = train_FedProx(*common_params, mu=mu)

    elif alg_name == 'FedAvgReg':
        _ = train_FedAvgReg(*common_params, mu, client_cls_counts_npy, global_dist)

    elif alg_name == 'FedSpeed':
        train_fed_speed(*common_params, alpha_coef=alpha_coef, sch_step=1, sch_gamma=1, rho=0.1, test_per=1)

    elif alg_name == 'centralized':
        train_centralized(*common_params)

if __name__ == '__main__':
    main()
