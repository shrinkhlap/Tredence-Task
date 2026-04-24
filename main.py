import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# --1: Prunable Linear Layer ---
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weights and biases
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Gate scores: initialized to a high value so gates start near 1 (Sigmoid(3) ~ 0.95)
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.gate_scores, 0.0) 

    def forward(self, x):
        # Transforming gate_scores to [0, 1] using Sigmoid
        gates = torch.sigmoid(self.gate_scores)
        
        # Element-wise multiplication to prune weights
        pruned_weights = self.weight * gates
        
        # Standard linear operation (weighted sum + bias): y = xW^T + b
        return nn.functional.linear(x, pruned_weights, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores)

# --- 2: The Neural Network Architecture ---
class SelfPruningNet(nn.Module):
    def __init__(self):
        super(SelfPruningNet, self).__init__()
        self.flatten = nn.Flatten()
        # MLP for CIFAR-10 (32x32x3 = 3072 input features)
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_all_gates(self):
        # Helper to collect all gate tensors for loss calculation
        return [self.fc1.get_gates(), self.fc2.get_gates(), self.fc3.get_gates()]

# --- 3: Training and Evaluation ---
def train_and_evaluate(lambd, epochs=20, device='cuda'):
    with mlflow.start_run():   
        
        # Log hyperparameter
        mlflow.log_param("lambda", lambd)
        mlflow.log_param("epochs", epochs)

        # Data Loading
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

        model = SelfPruningNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        print(f"\nTraining with Lambda={lambd}...")
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                class_loss = criterion(outputs, labels)
                
                #Sparsity Loss = sum(absolute values of gates) -> L1 Norm
                sparsity_loss = sum([torch.sum(torch.abs(g)) for g in model.get_all_gates()])
                
                # Total Loss = Classification Loss (CrossEntropyLoss) + lambda*(Sparsity Loss)
                total_loss = class_loss + lambd * sparsity_loss   

                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()

            epoch_loss = running_loss / len(trainloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

            # Log training loss per epoch
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)

        # --- Evaluation ---
        model.eval()
        correct = 0
        total = 0
        all_gate_vals = []

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            total_weights = 0
            pruned_weights = 0
            threshold = 1e-2

            for g in model.get_all_gates():
                flat_g = g.view(-1)
                all_gate_vals.extend(flat_g.cpu().tolist())
                total_weights += flat_g.numel()
                pruned_weights += torch.sum(flat_g < threshold).item()

        accuracy = 100 * correct / total
        sparsity_level = 100 * pruned_weights / total_weights

        # Log final metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("sparsity", sparsity_level)

        # Log model as well
        mlflow.pytorch.log_model(model, "model")

        return accuracy, sparsity_level, all_gate_vals
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lambdas = [1e-6, 1e-5, 1e-4]
    results = []

    for l in lambdas:
        acc, sparse, gate_vals = train_and_evaluate(l, epochs=20, device=device)
        results.append((l, acc, sparse, gate_vals))

    # Summary Table
    print("\n" + "="*30)
    print(f"{'Lambda':<10} | {'Accuracy (%)':<15} | {'Sparsity (%)':<15}")
    print("-" * 45)
    for l, acc, sparse, _ in results:
        print(f"{l:<10} | {acc:<15.2f} | {sparse:<15.2f}")
    
    # Plotting distribution for the best model
    best_gate_vals = results[1][3] # Results for 1e-5 (sample)
    plt.figure(figsize=(10, 6))
    plt.hist(best_gate_vals, bins=50, color='skyblue', edgecolor='black')
    plt.title(f"Gate Value Distribution (Lambda={results[1][0]})")
    plt.xlabel("Gate Value (Sigmoid Output)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    plt.show()

