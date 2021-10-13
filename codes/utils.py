import torch 

# This function computes the accuracy on the test dataset
def compute_accuracy(model, testloader):
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for pose, label in testloader:
            pose, label = pose.to(device), label.to(device)
            outputs = model(pose)
            _, predicted = torch.max(outputs.data, -1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    return correct / total