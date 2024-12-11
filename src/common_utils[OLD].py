import numpy as np

def partition_data(trn_x, trn_y, n_cls, n_client, rule_arg, clnt_data_list):
    # Generate Dirichlet class priors for each client
    cls_priors = np.random.dirichlet(alpha=[rule_arg]*n_cls, size=n_client)
    prior_cumsum = np.cumsum(cls_priors, axis=1)
    
    # Get indices of samples for each class
    idx_list = [np.where(trn_y==i)[0] for i in range(n_cls)]
    cls_amount = [len(idx_list[i]) for i in range(n_cls)]
    
    # Initialize each client data list
    clnt_x = [[] for _ in range(n_client)]
    clnt_y = [[] for _ in range(n_client)]
    
    total_samples = np.sum(clnt_data_list)
    assert total_samples != 0, 'Error! in clnt_data_list.'
    
    # Partition data across clients
    while total_samples > 0:
        curr_clnt = np.random.randint(n_client)
        
        # Skip if current client is full
        if clnt_data_list[curr_clnt] <= 0:
            continue
        
        clnt_data_list[curr_clnt] -= 1
        total_samples -= 1
        curr_prior = prior_cumsum[curr_clnt]
        
        while True:
            cls_label = np.argmax(np.random.uniform() <= curr_prior)
            
            if cls_amount[cls_label] <= 0:
                continue
            
            cls_amount[cls_label] -= 1
            
            clnt_x[curr_clnt].append(trn_x[idx_list[cls_label]])
            clnt_y[curr_clnt].append(trn_y[idx_list[cls_label]])
            
            break
        
        clnt_x = [np.array(client, dtype=np.float32) for clnt in clnt_x]
        clnt_y = [np.array(client, dtype=np.uint8).reshape(-1, 1) for client in clnt_y]
        
        return clnt_x, clnt_y