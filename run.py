import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pylab import *
from sklearn import metrics

# from stgcn import STGCN
from stgcn_tcn import STGCN

from utils_15d import generate_dataset, load_metr_la_data, get_normalized_adj

use_gpu = False
num_timesteps_input = 24
num_timesteps_output = 5

epochs = 50
batch_size = 50
# ------------------------------------------------------------------------cuda-version----------------------------------
# parser = argparse.ArgumentParser(description='STGCN')
# parser.add_argument('--enable-cuda', action='store_true',
#                     help='Enable CUDA')
# args = parser.parse_args()
# args.device = None
# print("torch.cuda.is_available(): ", torch.cuda.is_available())
# print("torch.cuda.device_count(): ", torch.cuda.device_count())
# args.device = torch.device('cuda')

# if args.enable_cuda and torch.cuda.is_available():
#     args.device = torch.device('cuda')
#     print("cuda")
# else:
#     args.device = torch.device('cpu')
#     print('cpu')
# -------------------------------------------------------------------UP---cuda-version----------------------------------
# ------------------------------------------------------------------DOWN--mps--version----------------------------------

# Check that MPS is available
print(torch.backends.mps.is_available())
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
else:
    print("If torch.backends.mps.is_available(): {}")
    parser = argparse.ArgumentParser(description='STGCN')
    args = parser.parse_args()
    args.device = torch.device("mps")


# ------------------------------------------------------------------  UP--mps--version----------------------------------



def train_epoch(training_input, training_target, batch_size):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input.shape[0])
    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        out = net(A_wave, X_batch)
        loss = loss_criterion(out, y_batch)
        # print("X_batch",X_batch)
        # print("y_batch",y_batch)
        # print(loss)

        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
        MSE = sum(epoch_training_losses) / len(epoch_training_losses)
        RMSE = np.sqrt(MSE)
    return RMSE


if __name__ == '__main__':
    torch.manual_seed(7)

    A, X, means, stds = load_metr_la_data()
    # A=torch.tensor(A,device=args.device)
    # X=torch.tensor(X,device=args.device)

    split_line1 = int(X.shape[2] * 0.6)
    split_line2 = int(X.shape[2] * 0.8)

    all_data = X[:, :, :]
    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]
    all_input, all_target = generate_dataset(all_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
    val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
    test_input, test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)

    A_wave = get_normalized_adj(A)
    # print(A_wave.shape[0],training_input.shape[3])
    A_wave = torch.from_numpy(A_wave)
    A_wave = A_wave.to(device=args.device)
    # print(training_input.shape)  :torch.Size([20549, 207, 12, 2])
    net = STGCN(A_wave.shape[0],
                training_input.shape[3],
                num_timesteps_input,
                num_timesteps_output).to(device=args.device)
    # print(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []
    test_losses = []

    train_RMSEs = []
    validation_RMSEs = []
    test_RMSEs = []

    train_MAEs = []
    validation_MAEs = []
    test_MAEs = []

    train_MAPEs = []
    validation_MAPEs = []
    test_MAPEs = []

    for epoch in range(epochs):
        print("epoch= ", epoch + 1,
              "--------------------------------------------------------------------------------------")
        loss = train_epoch(training_input, training_target,
                           batch_size=batch_size)
        training_losses.append(loss)

        # Run validation
        with torch.no_grad():
            net.eval()
            val_input = val_input.to(device=args.device)
            val_target = val_target.to(device=args.device)
            test_input = test_input.to(device=args.device)
            test_target = test_target.to(device=args.device)
            training_input = training_input.to(device=args.device)
            training_target = training_target.to(device=args.device)

            val_out = net(A_wave, val_input)
            train_out = net(A_wave, training_input)
            test_out = net(A_wave, test_input)

            # train部分
            train_out_unnormalized = train_out.detach().cpu().numpy() * stds[0] + means[0]
            training_target_unnormalized = training_target.detach().cpu().numpy() * stds[0] + means[0]

            train_rmse = np.sqrt(metrics.mean_squared_error(training_target_unnormalized[0], train_out_unnormalized[0]))
            train_MAE = metrics.mean_absolute_error(training_target_unnormalized[0], train_out_unnormalized[0])
            train_MAPE = metrics.mean_absolute_percentage_error(train_out_unnormalized[0], training_target_unnormalized[0])
            train_RMSEs.append(train_rmse)
            train_MAEs.append(train_MAE)
            train_MAPEs.append((train_MAPE))

            # validation 部分
            val_out_unnormalized = val_out.detach().cpu().numpy() * stds[0] + means[0]
            val_target_unnormalized = val_target.detach().cpu().numpy() * stds[0] + means[0]

            val_rmse = np.sqrt(metrics.mean_squared_error(val_target_unnormalized[0], val_out_unnormalized[0]))
            val_mae = metrics.mean_absolute_error(val_target_unnormalized[0], val_out_unnormalized[0])
            val_MAPE = metrics.mean_absolute_percentage_error(val_target_unnormalized[0], val_out_unnormalized[0])
            # val_mae = np.mean(np.absolute(val_out_unnormalized - val_target_unnormalized))
            # val_MAPE = np.mean(np.absolute((val_out_unnormalized - val_target_unnormalized) / val_out_unnormalized))
            validation_MAEs.append(val_mae)
            validation_MAPEs.append(val_MAPE)
            validation_RMSEs.append(val_rmse)

            # test 部分
            test_out_unnormalized = test_out.detach().cpu().numpy() * stds[0] + means[0]
            test_target_unnormalized = test_target.detach().cpu().numpy() * stds[0] + means[0]

            test_rmse = np.sqrt(metrics.mean_squared_error(test_target_unnormalized[0], test_out_unnormalized[0]))
            test_mae = metrics.mean_absolute_error(test_target_unnormalized[0], test_out_unnormalized[0])
            test_MAPE = metrics.mean_absolute_percentage_error(test_target_unnormalized[0], test_out_unnormalized[0])
            test_RMSEs.append(test_rmse)
            test_MAEs.append(test_mae)
            test_MAPEs.append(test_MAPE)

            val_out = None
            val_input = val_input.to(device=args.device)
            val_target = val_target.to(device=args.device)
            test_out = None
            test_input = test_input.to(device=args.device)
            test_target = test_target.to(device=args.device)

        # print("Training RMSE:   {}".format(train_RMSEs[-1]), "Training MAE:        {}".format(train_MAEs[-1]))
        # print("Validation RMSE: {}".format(validation_RMSEs[-1]), "Validation MAE: {}".format(validation_MAEs[-1]))
        print("Test RMSE:{}".format(test_RMSEs[-1]),"Test MAE: {}".format(test_MAEs[-1]),"Test MAPE: {}".format(test_MAPEs[-1]))
        print("SmallRMSE:",np.min(test_RMSEs), "SmallMAE:",np.min(test_MAEs), "SmallMAPE:",np.min(test_MAPEs))

        checkpoint_path = "checkpoints/"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        with open("checkpoints/losses.pk", "wb") as fd:
            pk.dump((training_losses, validation_losses, validation_MAEs), fd)
            torch.save(net.state_dict(), 'model_1y_{}-{}.pkl'.format(num_timesteps_input, num_timesteps_output))
    plt.plot(train_MAEs, label='Train MAE')
    plt.plot(validation_MAEs, label='Validation MAE')
    plt.plot(test_MAEs, label="Test MAE")
    plt.legend()
    plt.show()

    plt.plot(train_RMSEs, label="Training RMSE")
    plt.plot(validation_RMSEs, label="Validation RMSE")
    plt.plot(test_RMSEs, label="Test RMSE")
    plt.legend()
    plt.show()

    plt.plot(train_MAPEs, label='Training MAPE')
    plt.plot(validation_MAPEs, label='Validation MAPE')
    plt.plot(test_MAPEs, label='Test MAPE')
    plt.legend()
    plt.show()

    x_ori_plot = [i for i in range(num_timesteps_input + num_timesteps_output)]
    x_out_plt = [i for i in range(num_timesteps_input, num_timesteps_input + num_timesteps_output)]

    # print(all_data[0, 0, split_line2:split_line2 + num_timesteps_input + num_timesteps_output] * stds[0] + means[0])
    print(test_target_unnormalized[0, 25, :], test_out_unnormalized[0, 25, :])
    # print(np.min(test_RMSEs),np.min(test_MAEs),np.min(test_MAPEs))
    # plt.plot(x_ori_plot,
    #          all_data[0, 0, split_line2:split_line2 + num_timesteps_input + num_timesteps_output] * stds[0] + means[0])
    # plt.plot(x_out_plt, test_out_unnormalized[0, 0, :])
    # plt.legend(['Truth', "Predicted"])
    # res = []
    # res.append(
    #     all_data[0, 0, split_line2:split_line2 + num_timesteps_input + num_timesteps_output] * stds[0] + means[0])
    # res.append(test_out_unnormalized[0, 0, :])
    # np.save('predicted_result_d_{}->{}.npy'.format(num_timesteps_input, num_timesteps_output), res)
    print("Done!")
    plt.show()
