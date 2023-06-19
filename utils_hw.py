import torch
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import random

def train(
        model,
        config,
        train_loader,
        valid_loader=None,
        valid_epoch_interval=1,
        folder_name="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)  # config["lr"]=0.001
    best_valid_loss = 1e10
    print("start training...")

    for epoch_no in range(50):  # config["epochs"]=50
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )

            if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
                model.eval()
                avg_loss_valid = 0
                with torch.no_grad():
                    with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                        for batch_no, valid_batch in enumerate(it, start=1):
                            loss = model(valid_batch)
                            avg_loss_valid += loss.item()
                            it.set_postfix(
                                ordered_dict={
                                    "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                    "epoch": epoch_no,
                                },
                                refresh=False,
                            )
                if best_valid_loss > avg_loss_valid:
                    best_valid_loss = avg_loss_valid
                    print(
                        "\n best loss is updated to ",
                        avg_loss_valid / batch_no,
                        "at epoch_no",
                        epoch_no,
                    )
                    torch.save(model.state_dict(), folder_name + "/model.pth")


def generate(model, test_loader, folder_name):
    with torch.no_grad():
        model.eval()
        print("start imputing...")

        with tqdm(test_loader) as it:
            for batch_no, test_batch in enumerate(it):
                output = model.generated_sample(test_batch)
                generated_samples_batch, target_mask_batch = output

                # generated samples
                if batch_no == 0:
                    generated_samples = generated_samples_batch.detach().clone()
                else:
                    generated_samples = torch.cat([generated_samples, generated_samples_batch], dim=0)

                # target_mask
                if batch_no == 0:
                    target_mask = target_mask_batch.detach().clone()
                else:
                    target_mask = torch.cat([target_mask, target_mask_batch], dim=0)

                # observed data
                if batch_no == 0:
                    original_observed_data = test_batch['original_observed_data'].detach().clone()
                else:
                    original_observed_data = torch.cat([original_observed_data, test_batch['original_observed_data']], dim=0)

        torch.save(generated_samples, folder_name + '/generated_samples.pt')
        torch.save(target_mask, folder_name + '/target_mask.pt')
        torch.save(original_observed_data, folder_name + '/original_observed_data.pt')

    return generated_samples, original_observed_data, target_mask


def evaluate(generated_samples, original_observed_data, target_mask, mean, std):
    
    generated_samples = generated_samples.cpu()
    original_observed_data = original_observed_data.cpu()
    target_mask = target_mask.cpu()
    
    generated_samples = np.transpose(generated_samples, (0, 2, 1))
    generated_samples[0] = generated_samples[0]*std + mean
    generated_samples = np.transpose(generated_samples, (0, 2, 1))
    
    base = 2
    eps = 1e-6

    a = torch.sum(target_mask, dim=2)
    selected_pos = (a > 0)
    y_pred = generated_samples[selected_pos]
    y_true = original_observed_data[selected_pos]

    y_pred[y_pred < 1e-4] = -1e10
    y_pred = torch.softmax(y_pred, 1)

    y_true = y_true.numpy()
    y_pred = y_pred.numpy()

    log_op = np.log2(y_pred + eps) - np.log2(y_true + eps)
    mul_op = np.multiply(y_pred, log_op)
    sum_hist = np.sum(mul_op, axis=1)
    multi_factor = np.log2(base)
    sum_hist = sum_hist / multi_factor

    kl_average = np.mean(sum_hist)

    return kl_average


def generate_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
