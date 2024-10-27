import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

def main():
    # Initialize process group
    dist.init_process_group(backend='nccl')

    # Get the rank of the current process and set the device accordingly
    rank = dist.get_rank()
    print(f"Rank {rank} started")
    device = rank % torch.cuda.device_count()
    print(f"Rank {rank} got device {device}")

    # Create and move the model to the corresponding device
    model = SimpleModel().to(device)
    ddp_model = DDP(model, device_ids=[device])

    # Define a simple optimizer and loss function
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Generate dummy data for testing
    inputs = torch.randn(20, 10).to(device)
    targets = torch.randn(20, 10).to(device)

    # Perform a single forward-backward pass
    optimizer.zero_grad()
    outputs = ddp_model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()

    print(f"Rank {rank} | Loss: {loss.item()}")

    # Cleanup the process group
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
