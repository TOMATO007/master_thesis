import torch
import yaml
import random
import numpy as np
from main_model import CSDI_Physio
from utils_hw import train, generate, evaluate, generate_random_seed
from dataset_hw import get_dataloader

# load config
path = "./config/base.yaml"
with open(path, "r") as f:
    config = yaml.safe_load(f)
    
# data_num = [0, 1, 2, 3, 4]
# rm = [0.5, 0.6, 0.7, 0.8]
data_num = [0]
rm = [0.8] 

for j in data_num:
    for i in rm:
        
        # path
        folder_name = "./data/dataset_" + str(j) + "/rm" + str(i) 
        print(str(j) + "-" + str(i))
        
        # kl initial
        kl = 2
        
        # 1 data generation
        train_loader, valid_loader, test_loader = get_dataloader(batch_size=20, data_num=j, rm=i)
        torch.save(train_loader, folder_name + '/train_loader.pt')
        torch.save(valid_loader, folder_name + '/valid_loader.pt')
        torch.save(test_loader, folder_name + '/test_loader.pt')    
        
        # 2 training
        model = CSDI_Physio(config).to('cuda:0')
        train_loader = torch.load(folder_name + '/train_loader.pt')
        valid_loader = torch.load(folder_name + '/valid_loader.pt')
        train(model, config["train"], train_loader, valid_loader=valid_loader, 
              valid_epoch_interval=1, folder_name=folder_name)

        # 3 load model and generate sample
        model = CSDI_Physio(config).to('cuda:0')
        model.load_state_dict(torch.load(folder_name + '/model.pth'))
        model.eval()

        test_loader = torch.load(folder_name + '/test_loader.pt')
        generated_samples, original_observed_data, target_mask = \
            generate(model, test_loader, folder_name=folder_name)

        # 4 evaluation
        generated_samples = torch.load(folder_name + '/generated_samples.pt')
        original_observed_data = torch.load(folder_name + '/original_observed_data.pt')
        target_mask = torch.load(folder_name + '/target_mask.pt')
        mean = torch.load("./data/dataset_" + str(j) +'/mean.pt')
        std = torch.load("./data/dataset_" + str(j) +'/std.pt')                
        kl_average = evaluate(generated_samples, original_observed_data, target_mask, mean, std)
        print('current kl:', kl_average)

        if kl_average < kl:
            kl = kl_average
            torch.save(kl, folder_name + '/kl_minimum.pt')
            torch.save(forward_seed, folder_name + '/forward_seed.pt')
            torch.save(reverse_seed, folder_name + '/reverse_seed.pt')

        print('minimum kl:', kl)    
        print()
                
