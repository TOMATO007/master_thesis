import numpy as np
import torch
import torch.nn as nn
from diff_model import diff_CSDI
import torch.nn.functional as F


class CSDI_Physio(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = 'cuda:0'
        self.target_dim = 24
        self.emb_time_dim = config["model"]["timeemb"]  # 128
        self.emb_feature_dim = config["model"]["featureemb"]  # 16
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim + 1  # 128 + 16 + 1 = 145
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim)

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        input_dim = 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = torch.zeros(len(observed_data)).long().to(self.device)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            cut_length,
        )

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape  
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1) 
        feature_embed = self.embed_layer(torch.arange(self.target_dim).to(self.device))  
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1) 
        side_info = torch.cat([time_embed, feature_embed], dim=-1)
        side_info = side_info.permute(0, 3, 2, 1)
        side_mask = cond_mask.unsqueeze(1)          
        side_info = torch.cat([side_info, side_mask], dim=1)
        return side_info 

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        cond_obs = (cond_mask * observed_data).unsqueeze(1)  
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)  
        total_input = torch.cat([cond_obs, noisy_target], dim=1)  
        return total_input 

    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info):    
        B, K, L = observed_data.shape
        t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t] 
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)  
        predicted = self.diffmodel(total_input, side_info, t) 
        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss
    
    def impute(self, observed_data, cond_mask, side_info):
        current_sample = torch.randn_like(observed_data) 

        for t in range(self.num_steps - 1, -1, -1):  
            cond_obs = (cond_mask * observed_data).unsqueeze(1)  
            noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)  
            diff_input = torch.cat([cond_obs, noisy_target], dim=1) 
            predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))

            coeff1 = 1 / self.alpha_hat[t] ** 0.5
            coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
            current_sample = coeff1 * (current_sample - coeff2 * predicted)

            if t > 0:
                noise = torch.randn_like(current_sample)
                sigma = (
                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                ) ** 0.5
                current_sample += sigma * noise

        return current_sample

    def forward(self, batch):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            cut_length,
        ) = self.process_data(batch)

        cond_mask = gt_mask
        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss

        return loss_func(observed_data, cond_mask, observed_mask, side_info)

    def generated_sample(self, batch):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            side_info = self.get_side_info(observed_tp, cond_mask)
            samples = self.impute(observed_data, cond_mask, side_info)

        return samples, target_mask
