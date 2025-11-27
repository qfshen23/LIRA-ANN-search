import torch
import torch.nn as nn


class MLP_2_Input(nn.Module):
    '''
    two inputs
    1) vector, 2) distance to cluster center,
    '''
    def __init__(self, input_dim1, input_dim2, output_dim):
        super(MLP_2_Input, self).__init__()
        self.distance_net = nn.Sequential(
            nn.Linear(input_dim1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.vector_net = nn.Sequential(
            nn.Linear(input_dim2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x_dist, x_vec):
        out_dist = self.distance_net(x_dist)
        out_vec = self.vector_net(x_vec)
        combined = torch.cat((out_dist, out_vec), dim=1)
        output = self.fc(combined)
        
        return output
    
def model_train(model, train_loader, device, optimizer, criterion):
    model.train()
    total_loss = 0

    for inputs_dist, inputs_vec, targets in train_loader:
        inputs_dist, inputs_vec, targets = inputs_dist.to(device), inputs_vec.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs_dist, inputs_vec)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# def model_evaluate(model, test_loader, criterion, device):
#     '''
#     all_targets: the groundthruth 0/1 label of whether a bucket contains knn
#     all_predicts: the model output is converted to 0/1 label based on sigma threshold
#     all_outputs : the model output for query tuning with a threshold
#     '''
#     model.eval()
#     sigma = 0.5 # threshold for probing
#     with torch.no_grad():
#         all_outputs = [] # predicted probing possibility
#         all_predicts = [] # predicted 01 label with threshold
#         all_targets = [] # true 01 label
#         total_loss = 0

#         for inputs_dist, inputs_vec, targets in test_loader:
#             inputs_dist, inputs_vec, targets = inputs_dist.to(device), inputs_vec.to(device), targets.to(device)
#             outputs = model(inputs_dist, inputs_vec)
#             loss = criterion(outputs, targets)
#             total_loss += loss.item()
#             predicted = outputs > sigma
#             all_outputs.append(outputs)
#             all_predicts.append(predicted)
#             all_targets.append(targets)

#         all_outputs = torch.cat(all_outputs).cpu()
#         all_predicts = torch.cat(all_predicts).cpu()
#         all_targets = torch.cat(all_targets).cpu()
    
#     return all_targets, all_predicts, total_loss / len(test_loader), all_outputs

def model_evaluate(model, test_loader, criterion, device):
    """
    all_targets: ground truth 0/1 labels (bucket contains knn)
    all_predicts: model outputs converted to 0/1 labels based on sigma threshold
    all_outputs: raw model outputs (on CPU) for query tuning with thresholds
    """
    model.eval()
    sigma = 0.5  # threshold for probing

    all_outputs = []   # predicted probing probabilities (on CPU)
    all_predicts = []  # predicted 0/1 labels (on CPU)
    all_targets = []   # true 0/1 labels (on CPU)
    total_loss = 0.0

    with torch.no_grad():
        for inputs_dist, inputs_vec, targets in test_loader:
            # 1) 只把当前 batch 丢到 GPU
            inputs_dist = inputs_dist.to(device, non_blocking=True)
            inputs_vec  = inputs_vec.to(device, non_blocking=True)
            targets_gpu = targets.to(device, non_blocking=True)

            # 2) 前向 + loss 在 GPU 上算
            outputs_gpu = model(inputs_dist, inputs_vec)
            loss = criterion(outputs_gpu, targets_gpu)
            total_loss += loss.item()

            # 3) 立刻把结果搬回 CPU，再存进 list
            outputs_cpu = outputs_gpu.detach().cpu()
            targets_cpu = targets.detach().cpu()
            predicts_cpu = (outputs_cpu > sigma)

            all_outputs.append(outputs_cpu)
            all_predicts.append(predicts_cpu)
            all_targets.append(targets_cpu)

            # 4) 把 GPU 上不再需要的中间变量显式删掉，帮助释放显存
            del inputs_dist, inputs_vec, targets_gpu, outputs_gpu
            # 如果想更激进一点，可以偶尔调用：
            # torch.cuda.empty_cache()

    # 5) 在 CPU 上做 cat，不占用 GPU 显存
    all_outputs  = torch.cat(all_outputs, dim=0)
    all_predicts = torch.cat(all_predicts, dim=0)
    all_targets  = torch.cat(all_targets, dim=0)

    avg_loss = total_loss / len(test_loader)
    return all_targets, all_predicts, avg_loss, all_outputs


def model_infer(model, test_loader, device):
    '''
    all_predicts: the model output is converted to 0/1 label based on sigma threshold
    all_outputs : the model output for query tuning with a threshold
    '''
    model.eval()
    sigma = 0.5 # threshold for probing
    with torch.no_grad():
        all_outputs = [] # predicted probing possibility
        all_predicts = [] # predicted 01 label with threshold

        for inputs_dist, inputs_vec in test_loader:
            inputs_dist, inputs_vec = inputs_dist.to(device), inputs_vec.to(device)
            outputs = model(inputs_dist, inputs_vec)
            predicted = outputs > sigma
            all_outputs.append(outputs)
            all_predicts.append(predicted)

        all_outputs = torch.cat(all_outputs).cpu()
        all_predicts = torch.cat(all_predicts).cpu()
    
    return all_predicts, all_outputs