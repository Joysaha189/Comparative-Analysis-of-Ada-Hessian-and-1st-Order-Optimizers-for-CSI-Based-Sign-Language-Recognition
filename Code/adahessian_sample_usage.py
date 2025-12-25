


"""# **Importing Libraries**"""
import csv
import math
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets, transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.optimizer import Optimizer




"""# **Getting the Current Directory**"""
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)




"""# **Dataset Loader Function**"""
def getData(train_data_path, test_data_path, train_bs, test_bs):
    # Load the data from npz files
    train_data = np.load(train_data_path)
    test_data = np.load(test_data_path)

    # Assuming the .npz files contain arrays with keys 'data' and 'target'
    train_csi = train_data['data']
    train_labels = train_data['target']

    test_csi = test_data['data']
    test_labels = test_data['target']

    # Create a mapping dictionary for unique entries
    unique_entries = np.unique(train_labels)
    mapping = {entry: i for i, entry in enumerate(unique_entries)}
    num_classes = len(unique_entries)

    # Remap Y_train and Y_test using the mapping
    train_labels = np.array([mapping[entry] for entry in train_labels])
    test_labels = np.array([mapping[entry] for entry in test_labels])


    # Convert to PyTorch tensors and reshape
    train_csi = torch.tensor(train_csi, dtype=torch.float32)
    train_csi = train_csi.permute(0, 3, 2, 1)  # Permute to (N, C, H, W) format: (256, 3, 60, 200)
    train_labels = torch.tensor(train_labels, dtype=torch.long)

    test_csi = torch.tensor(test_csi, dtype=torch.float32)
    test_csi = test_csi.permute(0, 3, 2, 1)  # Permute to (N, C, H, W) format: (256, 3, 60, 200)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # Create TensorDatasets
    train_dataset = TensorDataset(train_csi, train_labels)
    test_dataset = TensorDataset(test_csi, test_labels)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_bs, shuffle=True)

    return train_loader, test_loader



"""# **Loading the data**"""
#give the path where the trainset and testset is located
#the modified train and testset has only 10 class from home dataset
#this may not exactly reproduce the result stated but can verify that the model works
train_loader, test_loader = getData(os.path.join(parent_dir, r'Dataset/Home_downlink/trainset_modified.npz'),
    os.path.join(parent_dir, r'Dataset/Home_downlink/testset_modified.npz'),
    train_bs=16,
    test_bs=16
)



"""# **Fine Unique Labels and Their Count**"""
all_train_labels = []
for data, labels in train_loader:
    all_train_labels.extend(labels.cpu().numpy())  # Move to CPU if using GPU
unique_labels, counts = np.unique(all_train_labels, return_counts=True)
total_classes= len(unique_labels)





"""# **Used CNN Network**"""
class CNN(nn.Module):

    def __init__(self, num_classes, input_shape=(3, 60, 200)):  # Input shape (channels, height, width)
        super(CNN, self).__init__()

        # Define layers
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.batch_norm = nn.BatchNorm2d(3, momentum=0.9, eps=1e-06)  # Batch Normalization after Conv
        self.relu = nn.ReLU()                # Activation after Batch Normalization
        self.pool = nn.AvgPool2d(kernel_size=3, stride=3, padding=1)
        self.dropout = nn.Dropout(p=0.7)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 20 * 67, num_classes)  # Adjusted linear layer input size


    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)  # Batch Normalization
        x = self.relu(x)        # ReLU activation
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x




"""# **Adahessian Class**"""
##This is the optimizer implementation
class Adahessian(Optimizer):
    """Implements AdaHessian algorithm."""
    def __init__(self, params, lr=0.15, betas=(0.9, 0.999), eps=1e-4,
                 weight_decay=5e-4, hessian_power=.5):

        # Parameter Setting
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError("Invalid Hessian power value: {}".format(hessian_power))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay= weight_decay, hessian_power=hessian_power)
        super(Adahessian, self).__init__(params, defaults)


    """# **Compute the hutchinson trace**"""
    def get_trace(self, params, grads):
        for i, grad in enumerate(grads):
            if grad.grad_fn is None:
                raise RuntimeError(f'Gradient tensor {i} does not have grad_fn. Make sure to call loss.backward(create_graph=True)')
        v = [2 * torch.randint_like(p, high=2) - 1 for p in params]
        hvs = torch.autograd.grad(grads, params, grad_outputs=v, only_inputs=True, retain_graph=True)
        hutchinson_trace = []
        for hv in hvs:
            param_size = hv.size()
            if len(param_size) <= 2:
                tmp_output = hv.abs()
            elif len(param_size) == 4:
                tmp_output = torch.mean(hv.abs(), dim=[2, 3], keepdim=True)
            else:
                tmp_output = hv.abs()
            hutchinson_trace.append(tmp_output)
        return hutchinson_trace

    """# **finding the loss using adahessian optimizer**"""
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        params = []
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    grads.append(p.grad)
        hut_traces = self.get_trace(params, grads)

        for p, grad, hut_trace in zip(params, grads, hut_traces):
            state = self.state[p]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)

            exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']
            beta1, beta2 = self.defaults['betas']
            state['step'] += 1

            exp_avg.mul_(beta1).add_(grad.detach(), alpha=1 - beta1)
            exp_hessian_diag_sq.mul_(beta2).addcmul_(hut_trace, hut_trace, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            k = self.defaults['hessian_power']
            denom = (exp_hessian_diag_sq.sqrt() ** k) / (math.sqrt(bias_correction2) ** k)
            denom.add_(self.defaults['eps'])

            p.data.add_(exp_avg / bias_correction1 / denom + self.defaults['weight_decay'] * p.data, alpha=-self.defaults['lr'])
        return loss




"""# **Test Function**"""
# Test function to evaluate model
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return correct / total



"""# **Utils**"""
device = torch.device("cpu")
model=CNN(num_classes=total_classes).to(device)
print(model)
print(f'Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
criterion = nn.CrossEntropyLoss()



"""# **Optimizer used in the experiment**"""
def get_optimizer(name, model_parameters, lr=1e-3, weight_decay=5e-4):
    optimizers = {
        'Adam': lambda: optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay),
        'Adamw': lambda: optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay),
        'RMSprop': lambda: optim.RMSprop(model_parameters, lr=lr, weight_decay=weight_decay),
        'SGD': lambda: optim.SGD(model_parameters, lr=lr, weight_decay=weight_decay, momentum=0.9),
        'Adagrad': lambda: optim.Adagrad(model_parameters, lr=lr, weight_decay=weight_decay),
        'Nadam': lambda: optim.NAdam(model_parameters, lr=lr, weight_decay=weight_decay),
        'Adahessian': lambda: Adahessian(model_parameters) #for Ada-hessian the paramters are defined in the main class
    }

    if name not in optimizers:
        raise ValueError(f"Optimizer '{name}' is not supported. Choose from: {list(optimizers.keys())}")

    return optimizers[name]()

"""# **Example Usage**"""
optimizer_name = 'Adahessian'  # Change this to test different optimizers
optimizer = get_optimizer(optimizer_name, model.parameters())
scheduler = lr_scheduler.MultiStepLR(optimizer, [80, 160, 240], gamma=0.1, last_epoch=-1)


"""# **Model Checking**"""
# Debug: Test one batch to ensure AdaHessian works
model.train()
device = torch.device("cpu")
model.to(device)
data, target = next(iter(train_loader))
data, target = data.to(device), target.to(device)
optimizer.zero_grad()
output = model(data)
loss = criterion(output, target)
loss.backward(create_graph=True)
optimizer.step()
print(f"Single batch processed successfully with '{optimizer_name}'")



# Create the 'checkpoint' directory if it doesn't exist
if not os.path.exists('checkpoint1'):
    os.makedirs('checkpoint1')



# Training loop
epochs=5     #change this to run for more epochs
best_acc = 0.0
train_losses = []  # To store training losses
train_accuracies = []  # To store training accuracies
test_accuracies = []  # To store test accuracies
import time
start_time= time.time()
for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch}")
    train_loss = 0.0
    total_samples = 0

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward(create_graph=True)  # AdaHessian requires create_graph=True
        optimizer.step()

        train_loss += loss.item() * target.size(0)
        total_samples += target.size(0)

    scheduler.step()
    train_loss /= total_samples
    train_losses.append(train_loss)  # Append training loss

    # # Calculate training accuracy
    train_acc = test(model, train_loader, device)  # Reuse test function for training accuracy
    train_accuracies.append(train_acc)  # Append training accuracy

    acc = test(model, test_loader, device)
    test_accuracies.append(acc)  # Append test accuracy

    print(f"Training Loss: {train_loss:.2f}")
    print(f"Training Accuracy: {train_acc * 100:.2f}%")  # Print training accuracy
    print(f"Test Accuracy: {acc * 100:.2f}%\n")

    # # Save checkpoint if best accuracy
    if acc > best_acc:
        best_acc = acc

    # uncomment this if you want the best model to save the best model
    #     torch.save({
    #         'epoch': epoch,
    #         'model': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #         'best_accuracy': best_acc,
    #     }, 'checkpoint1/netbest.pkl')
t_time= time.time()-start_time
print(f"Total training time: {t_time:.2f} seconds")
print(f"Best Test Accuracy: {best_acc * 100:.2f}%")


# uncomment this to save the results in a CSV file
# csv_file_path = os.path.join(parent_dir, r'Results/results(home_downlink_new5525).csv')
# with open(csv_file_path ,'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Test Accuracy'])
#     for epoch, train_loss, train_acc, test_acc in zip(range(1, epochs + 1), train_losses, train_accuracies, test_accuracies):
#         writer.writerow([epoch, f"{train_loss:.3f}", f"{train_acc:.3f}", f"{test_acc:.3f}"])