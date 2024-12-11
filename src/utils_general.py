from utils_libs import *
from utils_dataset import *
from utils_models import *
from utils_optimizer import speedOpt
from minimizers import SAM, ASAM

# class ClientDataLoader:
#     def __init__(self, pickle_file):
#         with open(pickle_file, 'rb') as f:
#             self.mapping_dict = pickle.load(f)
#             self.dataloaders = self.mapping_dict['dataloader']
            
#     def get_train_data(self, client_idx):
#         train_dl = self.dataloaders[client_idx]['train_dl_local']
#         return train_dl
    
#     def get_all_test_data(self):
#         all_test_dls = [len(self.dataloaders[client_idx]['test_dl_local']) for client_idx in self.dataloaders.keys()]
#         #assert False, all_test_dls
#         all_test_datasets = ConcatDataset([dl.dataset for dl in all_test_dls])
#         combined_test_dl = DataLoader(
#             all_test_datasets, 
#             batch_size=all_test_datasets[0].batch_size,
#             shuffle=False,
#             num_workers=0,
#         )
#         return combined_test_dl
    
#     def get_data_count(self, client_idx):
#         return 650
        # num_samples = 0
        # for _, batch_y, _ in self.dataloaders[client_idx]['train_dl_local']:
        #     num_samples += batch_y.shape[0]
            
        # return num_samples

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
            
        for key, value in config_data.items():
            setattr(self, key, value)
            
        self.device = torch.device(self.device if torch.cuda.is_available() else 'cpu')

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        return None
    
    def __setitem__(self, key, value):
        setattr(self, key, value)

device = Config('config.yml').device

max_norm = 10

# --- FedDisco based --- #
def get_distribution_difference(client_cls_counts, selected_clients, metric, hypo_distribution):
    local_distributions = client_cls_counts[np.array(selected_clients),:]
    local_distributions = local_distributions / local_distributions.sum(axis=1)[:,np.newaxis]
    
    if metric=='cosine':
        similarity_scores = local_distributions.dot(hypo_distribution)/ (np.linalg.norm(local_distributions, axis=1) * np.linalg.norm(hypo_distribution))
        difference = 1.0 - similarity_scores
    elif metric=='only_iid':
        similarity_scores = local_distributions.dot(hypo_distribution)/ (np.linalg.norm(local_distributions, axis=1) * np.linalg.norm(hypo_distribution))
        difference = np.where(similarity_scores>0.9, 0.01, float('inf'))
    elif metric=='l1':
        difference = np.linalg.norm(local_distributions-hypo_distribution, ord=1, axis=1)
    elif metric=='l2':
        difference = np.linalg.norm(local_distributions-hypo_distribution, axis=1)
    elif metric=='kl':
        # difference = torch.nn.functional.kl_div(local_distributions.log(), hypo_distribution, reduction='batchmean')
        difference = special.kl_div(local_distributions, hypo_distribution)
        difference = np.sum(difference, axis=1)

        difference = np.array([0 for _ in range(len(difference))]) if np.sum(difference) == 0 else difference / np.sum(difference)
    return difference.reshape(-1, 1)

def disco_weight_adjusting(old_weight, distribution_difference, a, b):
    weight_tmp = old_weight - a * distribution_difference + b

    if (np.sum(weight_tmp > 0) > 0):
        new_weight = np.copy(weight_tmp)
        new_weight[new_weight<0.0] = 0.0
    total_normalizer = sum([new_weight[r] for r in range(len(old_weight))])
    new_weight = [new_weight[r] / total_normalizer for r in range(len(old_weight))]
    return new_weight

def get_disco_adjusted_weights(client_cls_counts, weight_list, selected_clients, hypo_distribution, args):
    total_data_points = sum([weight_list[clnt] for clnt in selected_clients])
    fed_avg_freqs = np.array([weight_list[clnt] / total_data_points for clnt in selected_clients])
    distribution_difference = get_distribution_difference(client_cls_counts, selected_clients, args.disco_diff_measure, hypo_distribution)
    adjusted_weights = disco_weight_adjusting(fed_avg_freqs, distribution_difference, args.disco_a, args.disco_b)
    return np.array(adjusted_weights)

# --- Evaluate a NN model
def get_acc_loss(data_loader_list, model, dataset_name, w_decay = None):
    model.eval()
    model.to(device)
    acc_overall = 0
    loss_overall = 0
    n_tst =  0
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    #import pdb; pdb.set_trace()
    with torch.no_grad():
        for clnt_idx in range(len(data_loader_list)):
            data_loader = data_loader_list[clnt_idx]['test_dl_local']  
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                y_pred = model(batch_x)

                loss = loss_fn(y_pred, batch_y.reshape(-1).long())
                loss_overall += loss.item()

                # Accuracy calculation
                y_pred = y_pred.cpu().numpy()       
                y_pred = np.argmax(y_pred, axis=1).reshape(-1)
                batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
                acc_overall += np.sum(y_pred == batch_y)
                n_tst  += batch_x.shape[0]
          
    #n_tst = sum(len(batch) for _, batch_y in data_loader)
    loss_overall /= n_tst
    acc_overall /= n_tst

    if w_decay != None:
        # Add L2 loss
        params = get_mdl_params([model], n_par=None)
        loss_overall += w_decay/2 * np.sum(params * params)
        
    model.train()
    return loss_overall, acc_overall

def get_per_class_acc_loss(data_x, data_y, model, dataset_name, w_decay = None):
    acc_overall = 0; loss_overall = 0;
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    batch_size = min(128, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model.eval(); model = model.to(device)
    raw_scores = []
    unique_labels = np.unique(data_y)
    acc_arr = np.zeros(len(unique_labels))
    acc_loss = np.zeros(len(unique_labels))
    ref_labels = np.zeros(len(unique_labels))
    print("unique_labels:",unique_labels)
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst/batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)
            
            batch_y = batch_y.cpu().numpy()
            y_pred = y_pred.cpu().numpy()

            for i in unique_labels:
                des_idx = np.nonzero(batch_y == i)[0]
                #print("des_idx:",des_idx)
                #print("y_pred:",y_pred)
                y_pred_desidx = y_pred[des_idx].copy()
                batch_y_desidx = batch_y[des_idx].copy()
                loss = loss_fn(torch.Tensor(y_pred), torch.Tensor(batch_y).reshape(-1).long())
                acc_loss[i] += loss
                # Accuracy calculation
                #y_pred_desidx = y_pred_desidx.cpu().numpy()            
                y_pred_desidx = np.argmax(y_pred_desidx, axis=1).reshape(-1)
                #batch_y_desidx = batch_y_desidx.cpu().numpy().reshape(-1).astype(np.int32)
                batch_y_desidx = batch_y_desidx.reshape(-1).astype(np.int32)
                
                #print("batch_y_desidx:",batch_y_desidx.shape)
                #print("y_pred_desidx:",y_pred_desidx.shape)
                batch_correct = np.sum(y_pred_desidx == batch_y_desidx)
                acc_arr[i] += batch_correct
                ref_labels[i]+= len(batch_y_desidx) 

    loss_overall /= n_tst
    acc_loss = acc_loss/ref_labels
    acc_arr = acc_arr/ref_labels
    return acc_loss,acc_arr

def get_true_pred_scores(data_x, data_y, model, dataset_name, w_decay = None):
    acc_overall = 0; loss_overall = 0;
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    batch_size = min(128, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model.eval(); model = model.to(device)
    raw_scores = []
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst/batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)
            
            #print("y_pred:",y_pred,"\t","batch_y:",batch_y)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_overall += loss.item()
            # Accuracy calculation
            y_pred_raw = y_pred.cpu().numpy()            
            y_pred = np.argmax(y_pred_raw, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct
            ## get confident scores ##
            y_pred_scores = np.max(y_pred_raw, axis=1).reshape(-1)
            raw_scores.append(y_pred_scores[y_pred == batch_y])

    loss_overall /= n_tst
    raw_scores = np.concatenate(raw_scores,axis = 0)

    model.train()
    return raw_scores

# --- Helper functions

def set_client_from_params(mdl, params):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx+length].reshape(weights.shape)).to(device))
        idx += length
    
    mdl.load_state_dict(dict_param)
    return mdl


def get_mdl_params(model_list, n_par=None):
    
    if n_par==None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))
    
    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)

# --- Train functions

def train_model(model, train_loader, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name):
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train()
    model.to(device)
    
    for e in range(epoch):
        for batch_x, batch_y, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss = loss / list(batch_y.size())[0]
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()

        if (e+1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(train_loader, model, dataset_name, weight_decay)
            print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f" %(e+1, acc_trn, loss_trn))
            model.train()
        
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

# def train_scaffold_mdl(model, model_func, state_params_diff, trn_x, trn_y, learning_rate, batch_size, n_minibatch, print_per, weight_decay, dataset_name):
#     n_trn = trn_x.shape[0]
#     trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
#     loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#     model.train(); model = model.to(device)
    
#     n_par = get_mdl_params([model_func()]).shape[1]
#     n_iter_per_epoch = int(np.ceil(n_trn/batch_size))
#     epoch = np.ceil(n_minibatch / n_iter_per_epoch).astype(np.int64)
#     count_step = 0
#     is_done = False
    
#     step_loss = 0; n_data_step = 0
#     for e in range(epoch):
#         # Training
#         if is_done:
#             break
#         trn_gen_iter = trn_gen.__iter__()
#         for i in range(int(np.ceil(n_trn/batch_size))):
#             count_step += 1
#             if count_step > n_minibatch:
#                 is_done = True
#                 break
#             batch_x, batch_y = trn_gen_iter.__next__()
#             batch_x = batch_x.to(device)
#             batch_y = batch_y.to(device)
            
#             y_pred = model(batch_x)
            
#             ## Get f_i estimate 
#             loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
#             loss_f_i = loss_f_i / list(batch_y.size())[0]
            
#             # Get linear penalty on the current parameter estimates
#             local_par_list = None
#             for param in model.parameters():
#                 if not isinstance(local_par_list, torch.Tensor):
#                 # Initially nothing to concatenate
#                     local_par_list = param.reshape(-1)
#                 else:
#                     local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
#             loss_algo = torch.sum(local_par_list * state_params_diff)
#             loss = loss_f_i + loss_algo

#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
#             optimizer.step()
#             step_loss += loss.item() * list(batch_y.size())[0]; n_data_step += list(batch_y.size())[0]

#             if (count_step) % print_per == 0:
#                 step_loss /= n_data_step
#                 if weight_decay != None:
#                     # Add L2 loss to complete f_i
#                     params = get_mdl_params([model], n_par)
#                     step_loss += (weight_decay)/2 * np.sum(params * params)
#                 print("Step %3d, Training Loss: %.4f" %(count_step, step_loss))
#                 step_loss = 0; n_data_step = 0
#                 model.train()
    
#     # Freeze model
#     for params in model.parameters():
#         params.requires_grad = False
#     model.eval()
            
#     return model

def train_feddyn_mdl(model,model_func, alpha_coef, avg_mdl_param, local_grad_vector, train_loader, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name):
    #print("FedDyn no reg")  
    # n_trn = trn_x.shape[0]
    n_trn = len(train_loader.dataset)
    # trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef+weight_decay)
    model.train()
    model.to(device)
    
    n_par = get_mdl_params([model_func()]).shape[1]
    
    for e in range(epoch):
        # Training
        epoch_loss = 0
        # trn_gen_iter = trn_gen.__iter__()
        # for i in range(int(np.ceil(n_trn/batch_size))):
            # batch_x, batch_y = trn_gen_iter.__next__()
        for batch_x, batch_y, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_x)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]
            
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = alpha_coef.to(local_par_list.device) * torch.sum(local_par_list * (-avg_mdl_param + local_grad_vector))
            #loss_algo = alpha_coef * torch.sum(local_par_list * (local_grad_vector))
            loss = loss_f_i + loss_algo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e+1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef+weight_decay)/2 * np.sum(params * params)
            print("Epoch %3d, Training Loss: %.4f" %(e+1, epoch_loss))
            model.train()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

def train_feddyn_mdl_reg(model,cld_model, model_func, alpha_coef, avg_mdl_param, local_grad_vector, train_loader, learning_rate, batch_size, epoch, print_per, 
        weight_decay, dataset_name,args):
    #print("Fed Dyn with reg")
    # n_trn = trn_x.shape[0]
    n_trn = len(train_loader.dataset)
    local_model_copy = copy.deepcopy(model)
    local_model_copy.eval()
    # model.load_state_dict(copy.deepcopy(dict(cld_model.named_parameters())))
    model.load_state_dict(cld_model.state_dict())
    for params in model.parameters():
        params.requires_grad = True

                # Scale down
    # trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    #bce_loss = torch.nn.BCEWithLogitsLoss(reduction = 'mean') 
    model.eval()

    srvr_model_copy = copy.deepcopy(model)
    for param in srvr_model_copy.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef+weight_decay)
    model.train()
    model.to(device)
    n_par = get_mdl_params([model_func()]).shape[1]
    
    trn_y = []
    with torch.no_grad():
        for batch_x, batch_y,_ in train_loader:
            trn_y.append(batch_y.reshape(-1).long())
            
    trn_y = torch.cat(trn_y,dim=0) 
    
    client_labels = torch.from_numpy(np.squeeze(trn_y))
    
    label_count = torch.bincount(client_labels).to(device)
    label_probs = (1.0*label_count)/torch.sum(label_count)

    for e in range(epoch):
        # Training
        epoch_loss = 0
        # trn_gen_iter = trn_gen.__iter__()
        # for i in range(int(np.ceil(n_trn/batch_size))):
        for batch_x, batch_y, _ in train_loader:
            # batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
           
            label_weights = -torch.log(label_probs[batch_y.reshape(-1).long()])

            y_pred = model(batch_x)
            
            with torch.no_grad():
                spred = srvr_model_copy(batch_x)
                lpred = local_model_copy(batch_x)

            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]
            
            loss_mean_algo = 0.0
            s_pred_temp = F.softmax(spred/args.temp, dim=1)
            y_pred_temp = F.softmax(y_pred/args.temp, dim=1)
            
            s_pred_notemp = F.softmax(spred, dim=1)
            l_pred_notemp = F.softmax(lpred, dim=1)
            
            w_kl = torch.sum(s_pred_notemp * torch.log(s_pred_notemp/l_pred_notemp),axis = 1)
            KL_loss = torch.sum(s_pred_temp * torch.log(s_pred_temp/y_pred_temp),axis = 1)
            breg_div = torch.sum(-torch.log(y_pred_temp),axis =1) + torch.sum(torch.log(s_pred_temp),axis=1) + torch.sum((1/s_pred_temp)*(y_pred_temp - s_pred_temp),axis=1)
            #print("KL_loss:",KL_loss.shape)
            server_entropy = -1.0*torch.sum(s_pred_temp * torch.log(s_pred_temp),axis = 1)
            
            
            label_imbalance_loss = torch.exp(args.dist_beta * label_weights)

            if args.entropy_flag == 0:
                distill_weights = ((1 - torch.exp(-w_kl)) ** args.dist_beta_kl) * label_imbalance_loss
            else:
                distill_weights = (torch.exp(-server_entropy)** args.dist_beta_kl) * label_imbalance_loss
            
            distill_weights = distill_weights/torch.sum(distill_weights)
            
            if args.breg_div == 0:
                final_reg_loss = KL_loss
            else:
                final_reg_loss = breg_div

            if args.uniform_distill == 0:
                distill_loss = torch.sum(distill_weights*final_reg_loss)
            else:
                distill_loss = torch.mean(final_reg_loss)
            
            loss = loss_f_i + args.lamda * distill_loss

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
              
            loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + local_grad_vector))
            
            if e == 0 and i == 0:
                loss = loss_f_i + loss_algo
            else:
                loss = loss_f_i + loss_algo +  (args.lamda) * distill_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e+1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (alpha_coef+weight_decay)/2 * np.sum(params * params)
            print("Epoch %3d, Training Loss: %.4f" %(e+1, epoch_loss))
            model.train()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

def train_fedprox_mdl(model, avg_model,avg_model_param_, args,mu, train_loader, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name):
    # n_trn = trn_x.shape[0]
    n_trn = len(train_loader.dataset)
    local_model_copy = copy.deepcopy(model)
    local_model_copy.eval()
    # model.load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))
    model.load_state_dict(avg_model.state_dict())
    for params in model.parameters():
        params.requires_grad = True

    # trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    model.eval()

    srvr_model_copy = copy.deepcopy(model)
    for param in srvr_model_copy.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(device)
    n_par = len(avg_model_param_)
    
    trn_y = []
    with torch.no_grad():
        for batch_x, batch_y,_ in train_loader:
            trn_y.append(batch_y.reshape(-1).long())
            
    trn_y = torch.cat(trn_y,dim=0) 
    
    client_labels = torch.from_numpy(np.squeeze(trn_y))
    
    label_count = torch.bincount(client_labels).to(device)
    label_probs = (1.0*label_count)/torch.sum(label_count)
    
    for e in range(epoch):
        # Training
        epoch_loss = 0
        # trn_gen_iter = trn_gen.__iter__()
        # for i in range(int(np.ceil(n_trn/batch_size))):
        for batch_x, batch_y, _ in train_loader:
            # batch_x, batch_y = trn_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            label_weights = -torch.log(label_probs[batch_y.reshape(-1).long()])
            y_pred = model(batch_x)
            
            ## Get f_i estimate 
            loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_f_i = loss_f_i / list(batch_y.size())[0]
            
            with torch.no_grad():
                spred = srvr_model_copy(batch_x)
                lpred = local_model_copy(batch_x)

            s_pred_temp = F.softmax(spred/args.temp, dim=1)
            y_pred_temp = F.softmax(y_pred/args.temp, dim=1)
            
            s_pred_notemp = F.softmax(spred, dim=1)
            l_pred_notemp = F.softmax(lpred, dim=1)
            
            w_kl = torch.sum(s_pred_notemp * torch.log(s_pred_notemp/l_pred_notemp),axis = 1)
            KL_loss = torch.sum(s_pred_temp * torch.log(s_pred_temp/y_pred_temp),axis = 1)
            breg_div = torch.sum(-torch.log(y_pred_temp),axis =1) + torch.sum(torch.log(s_pred_temp),axis=1) + torch.sum((1/s_pred_temp)*(y_pred_temp - s_pred_temp),axis=1)
            #print("KL_loss:",KL_loss.shape)
            server_entropy = -1.0*torch.sum(s_pred_temp * torch.log(s_pred_temp),axis = 1)
            label_imbalance_loss = torch.exp(args.dist_beta * label_weights)

            if args.entropy_flag == 0:
                distill_weights = ((1 - torch.exp(-w_kl)) ** args.dist_beta_kl) * label_imbalance_loss
            else:
                distill_weights = (torch.exp(-server_entropy)** args.dist_beta_kl) * label_imbalance_loss
            
            distill_weights = distill_weights/torch.sum(distill_weights)
            
            if args.breg_div == 0:
                final_reg_loss = KL_loss
            else:
                final_reg_loss = breg_div

            if args.uniform_distill == 0:
                distill_loss = torch.sum(distill_weights*final_reg_loss)
            else:
                distill_loss = torch.mean(final_reg_loss)


            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = mu/2 * torch.sum(local_par_list * local_par_list)
            loss_algo += -mu * torch.sum(local_par_list * avg_model_param_)
            loss = loss_f_i + loss_algo
            
            if args.add_reg > 0:
                loss = loss + (args.lamda*distill_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]

        if (e+1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += weight_decay/2 * np.sum(params * params)
            
            print("Epoch %3d, Training Loss: %.4f" %(e+1, epoch_loss))
            model.train()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

def train_fedavgreg_mdl(model,avg_model,avg_model_param_, mu, trn_loader, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name,args):
    n_trn = len(trn_loader.dataset)
    
    local_model_copy = copy.deepcopy(model)    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    trn_y = []
    with torch.no_grad():
        for batch_x, batch_y,_ in trn_loader:
            trn_y.append(batch_y.reshape(-1).long())

    trn_y = torch.cat(trn_y,dim=0)
    
    #bin_count = torch.bincount(torch.from_numpy(trn_y.reshape(-1)).long())
    bin_count = torch.bincount(trn_y.reshape(-1).long())
    #print("bin_count:",bin_count)
    label_probs = (1.0*bin_count)/torch.sum(bin_count)

    model.eval()
    avg_model.eval()
    local_model_copy.eval()
    
    srvr_model_copy = copy.deepcopy(model)
    for param in srvr_model_copy.parameters():
        param.requires_grad = False
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(device)

    client_labels = trn_y #torch.from_numpy(np.squeeze(trn_y))
    label_count = torch.bincount(client_labels).to(device)
    
    wsm_weights = torch.zeros(100).to(device)
    
    label_probs = (1.0*label_count)/torch.sum(label_count)
    for i in range(label_probs.shape[0]):
        wsm_weights[i] = label_probs[i]*100

    #print("max_label_idx:",torch.max(client_labels))
    #print("wsm_weights:",torch.sum(wsm_weights))
    #exit()

    batch_index = torch.arange(batch_size).to(device)
    n_par = len(avg_model_param_)
    #torch_inf = torch.tensor(float('inf')).to(device)

    for e in range(epoch):
        # Training
        epoch_loss = 0
        for batch_x, batch_y, _ in trn_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            label_weights = -torch.log(label_probs[batch_y.reshape(-1).long()])
            y_pred = model(batch_x)

            if args.wsm == 1:
                true_preds_ce = y_pred[torch.arange(batch_x.shape[0]),batch_y.reshape(-1).long()]
                scale_y_pred = torch.exp(y_pred)*wsm_weights
                negval = torch.log(torch.sum(scale_y_pred,dim=1))
                loss_f_i = -1.0*torch.mean(true_preds_ce - negval)
                #print("loss_f_i:",loss_f_i)
                 
            else:
                loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
                loss_f_i = loss_f_i / list(batch_y.size())[0]

            with torch.no_grad():
                spred = srvr_model_copy(batch_x)
                lpred = local_model_copy(batch_x)
                        
            loss_mean_algo = 0.0
            num_class = spred.shape[1]
            
            if args.ntd == 1:
                s_pred_copy = spred.clone()
                y_pred_copy = y_pred.clone()
                #print("batch_y:",batch_y[:10])
                mask_ = torch.ones_like(s_pred_copy).scatter_(1,batch_y.long(),0.)
                s_pred_update = s_pred_copy[mask_.bool()].view(batch_x.shape[0],num_class-1)
                y_pred_update = y_pred_copy[mask_.bool()].view(batch_x.shape[0],num_class-1)
                s_pred_temp = F.softmax((s_pred_update-torch.max(s_pred_update))/args.temp, dim=1)
                y_pred_temp = F.softmax((y_pred_update- torch.max(y_pred_update))/args.temp, dim=1)
                KL_loss = torch.sum(s_pred_temp * torch.log(s_pred_temp/y_pred_temp),axis = 1) 
            else:
                s_pred_temp = F.softmax(spred/args.temp, dim=1)
                y_pred_temp = F.softmax(y_pred/args.temp, dim=1)
                KL_loss = torch.sum(s_pred_temp * torch.log(s_pred_temp/y_pred_temp),axis = 1)
                rKL_loss = torch.sum(y_pred_temp * torch.log(y_pred_temp/s_pred_temp),axis = 1)

            s_pred_notemp = F.softmax(spred, dim=1)
            l_pred_notemp = F.softmax(lpred, dim=1)
            
            server_entropy = -1.0*torch.sum(s_pred_notemp * torch.log(s_pred_notemp),axis = 1)
            true_preds = s_pred_notemp[torch.arange(batch_x.shape[0]),batch_y.reshape(-1).long()]
            
            w_KL =  torch.sum(s_pred_notemp * torch.log(s_pred_notemp/l_pred_notemp),axis = 1)
            label_imbalance_loss = torch.exp(args.dist_beta * label_weights)
            margin_loss = torch.exp(-args.dist_beta_kl*w_KL)
            
            if args.entropy_flag == 0:
                distill_weights = ((1 - torch.exp(-w_KL)) ** args.dist_beta_kl) * label_imbalance_loss
            else:
                distill_weights = (torch.exp(-server_entropy)** args.dist_beta_kl) * label_imbalance_loss
            
            distill_weights = distill_weights/torch.sum(distill_weights)
                        
            final_reg_loss = KL_loss
            
            if args.uniform_distill == 0:
                distill_loss = torch.sum(distill_weights*final_reg_loss)
            else:
                distill_loss = torch.mean(final_reg_loss)

            #distill_loss = torch.mean(torch.linalg.norm(y_pred-spred,dim=1)**2)
            
            loss = loss_f_i + args.lamda * distill_loss

            #print("args.lamda:",args.lamda)
            #exit()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer.step()
            epoch_loss += loss.item() * list(batch_y.size())[0]
        
        #print("done one epoch")
        #exit()
        if (e+1) % print_per == 0:
            epoch_loss /= n_trn
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                epoch_loss += (weight_decay)/2 * np.sum(params * params)
            
            print("Epoch %3d, Training Loss: %.4f" %(e+1, epoch_loss))
            model.train()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
        
    model.eval()
    return model

# def train_central_model(data_obj,model, trn_x, trn_y, learning_rate, batch_size, epoch, print_per, lr_decay_per_round,weight_decay, dataset_name,args):
#     n_trn = trn_x.shape[0]
#     trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True) 
#     loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_per_round)
#     model.train(); model = model.to(device)

#     trn_perf_all = np.zeros((epoch, 2))
#     tst_perf_all = np.zeros((epoch, 2))
#     print("lr_decay_per_round:",lr_decay_per_round)
#     print("epoch:",epoch) 
#     for e in range(epoch):
#         # Training
        
#         trn_gen_iter = trn_gen.__iter__()
#         for i in range(int(np.ceil(n_trn/batch_size))):
#             batch_x, batch_y = trn_gen_iter.__next__()
#             batch_x = batch_x.to(device)
#             batch_y = batch_y.to(device)
            
#             y_pred = model(batch_x)

#             loss = loss_fn(y_pred, batch_y.reshape(-1).long())
#             loss = loss / list(batch_y.size())[0]


#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
#             optimizer.step()
        
#         scheduler.step()        
#         loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y, model, data_obj.dataset)
#         tst_perf_all[e] = [loss_tst, acc_tst]
#         loss_trn, acc_trn = get_acc_loss(trn_x, trn_y,model, data_obj.dataset)
#         trn_perf_all[e] = [loss_trn, acc_trn]

#         print("e:",e,"\t","trn_loss:",loss_trn,"\t","tst_loss:",loss_tst,"\t","acc_trn:",acc_trn,"\t","acc_tst:",acc_tst)
    
#     torch.save(model.state_dict(), 'centralized_reg.pth')
#     for params in model.parameters():
#         params.requires_grad = False
#     model.eval()


#     return trn_perf_all,tst_perf_all

def train_model_speed(args,all_model,model,model_func, alpha_coef, avg_mdl_param, hist_params_diff, train_loader,
                      learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name, sch_step, sch_gamma, 
                      rho, data_loader_dict, print_verbose=False):
    n_train = len(train_loader.dataset)
    
    train_y = []
    with torch.no_grad():
        for batch_x, batch_y,_ in train_loader:
            train_y.append(batch_y.reshape(-1).long())
            
    train_y = torch.cat(train_y, dim=0)
    
    local_model_copy = copy.deepcopy(model)
    model.load_state_dict(all_model.state_dict())
    # model.load_state_dict(copy.deepcopy(dict(all_model.named_parameters())))
    # train_gen = data.DataLoader(Dataset(train_x, train_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
    #                             shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    model.eval()
    
    srvr_model_copy = copy.deepcopy(model)
    for param in srvr_model_copy.parameters():
        param.requires_grad = False
    srvr_model_copy.eval()
    local_model_copy.eval()
    base_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef + weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(base_optimizer, step_size=sch_step, gamma=sch_gamma)
    
    for params in model.parameters():    
        params.requires_grad = True
        
    model.train()
    model.to(device)

    n_par = get_mdl_params([model_func()]).shape[1]
    
    client_labels = torch.from_numpy(np.squeeze(train_y)).long()
    
    label_count = torch.bincount(client_labels).to(device)
    label_probs = (1.0*label_count)/torch.sum(label_count)

    optimizer = speedOpt(model.parameters(), base_optimizer, rho=rho, beta=1.0, gamma=1.0, adaptive=False,
                         nograd_cutoff=0.05)

    for e in range(epoch):
        # Training
        # train_gen_iter = train_gen.__iter__()
        # for i in range(int(np.ceil(n_train / batch_size))):
        i = 0
        for batch_x, batch_y, _ in train_loader:
            i += 1
            # batch_x, batch_y = train_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_y = batch_y.reshape(-1).long()

            label_weights = -torch.log(label_probs[batch_y.reshape(-1).long()])

            with torch.no_grad():
                spred = srvr_model_copy(batch_x)
                lpred = local_model_copy(batch_x)
                
            def defined_backward(loss):
                loss.backward()

            paras = [batch_x, batch_y, loss_fn, model, defined_backward, spred,lpred, label_weights, i, args]
            optimizer.paras = paras
            optimizer.step()

            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + hist_params_diff))
            # loss = loss_f_i + loss_algo
            loss = loss_algo

            ### 
            # base_optimizer.zero_grad()
            loss.sum().backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)
            base_optimizer.step()

        if print_verbose and (e + 1) % print_per == 0:
            loss_train, acc_train = get_acc_loss(data_loader_dict, model, dataset_name, weight_decay)
            print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f" % (
                e + 1, acc_train, loss_train, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model



#def compute_gen_entropy(p_in,q_in,p_val):
#
#   gen_ent =  1 - (1/(p_val-1))*torch.sum(p_in**p_val,axis = 1)
#   gen_ce1 =  (q_in**p_val) * (p_in/q_in)
#   gen_ce = 1 -  (1/(p_val-1))*(torch.sum(gen_ce1,axis = 1))
#
#   return gen_ent + gen_ce

def train_model_sam(model, train_loader, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name, args, data_loader_dict):
    # n_trn = len(train_loader.dataset)
    # trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
    #                           shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #minimizer = SAM(optimizer, model, 0.05, 0.0)
    if args.rule == 'iid' and args.dataset_name == 'TinyImageNet':
        if args.mu_mean > 0:
            minimizer = SAM(optimizer, model, 0.01, 0.0) ##### iid
        else:
            minimizer = SAM(optimizer, model, 0.03, 0.0) ##### iid
    else:
        minimizer = SAM(optimizer, model,args.search_param, 0.0)
    
    #print("search_param:",args.search_param)
    model.train();
    model = model.to(device)

    for e in range(epoch):
        # Training

        # trn_gen_iter = trn_gen.__iter__()
        # for i in range(int(np.ceil(n_trn / batch_size))):
            # batch_x, batch_y = trn_gen_iter.__next__()
        for batch_x, batch_y, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred, var_val, _ = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            # loss = loss / list(batch_y.size())[0]
            loss_var_reg = 0.0
            for val in var_val:
                loss_var_reg += torch.mean(val)
            
            loss += (args.mu_mean/2)*loss_var_reg
            #optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
            minimizer.ascent_step()

            #loss_fn(model(batch_x)[0], batch_y.reshape(-1).long()).backward()
            y_pred,var_val,_ = model(batch_x) 
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            
            loss_var_reg = 0.0
            for val in var_val:
                loss_var_reg += torch.mean(val)
            
            loss += (args.mu_mean/2)*loss_var_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
            minimizer.descent_step()

        if (e + 1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(data_loader_dict, model, dataset_name, weight_decay)
            print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f" % (e + 1, acc_trn, loss_trn))
            model.train()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model

def train_model_asam(model, train_loader, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name, args, data_loader_dict):
    # n_trn = trn_x.shape[0]
    # trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
    #                           shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    if args.rule == 'iid' and (args.dataset_name== 'TinyImageNet' or args.mu_mean > 0.0):
        minimizer = ASAM(optimizer, model, 0.1, 0.2)
    else: 
        minimizer = ASAM(optimizer, model, 0.5, 0.2)
        #minimizer = ASAM(optimizer, model, 0.1, 0.2)

    model.train();
    model = model.to(device)

    for e in range(epoch):
        # Training
        # trn_gen_iter = trn_gen.__iter__()
        # for i in range(int(np.ceil(n_trn / batch_size))):
            # batch_x, batch_y = trn_gen_iter.__next__()
        for batch_x, batch_y, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred,var_val, _ = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            # loss = loss / list(batch_y.size())[0]
            loss_var_reg = 0.0
            for val in var_val:
                loss_var_reg += torch.mean(val)
            
            loss += (args.mu_mean/2)*loss_var_reg
            #optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
            minimizer.ascent_step()
            
            y_pred,var_val,_ = model(batch_x) 
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_var_reg = 0.0
            for val in var_val:
                loss_var_reg += torch.mean(val)
            
            loss += (args.mu_mean/2)*loss_var_reg
            loss.backward()
            #loss_fn(model(batch_x)[0], batch_y.reshape(-1).long()).backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
            minimizer.descent_step()
            # optimizer.step()

        if (e + 1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(data_loader_dict, model, dataset_name, weight_decay)
            print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f" % (e + 1, acc_trn, loss_trn))
            model.train()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model

