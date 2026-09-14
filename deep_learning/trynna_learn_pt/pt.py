# %%
import torch

# %%
x = torch.tensor([[1,2] , [3,4]] , dtype=torch.float32)
print(x.dtype)
# %%

zeros = torch.zeros((3,3))
print(zeros)
print(zeros.dtype)
# %%
rand = torch.randn((2,4))
print(rand)
# %%
device = "cuda" if torch.cuda.is_available() else "cpu" 
print(device)
# %%

w = torch.tensor([2.0] , requires_grad = True) 
b = torch.tensor([1.0] , requires_grad = True)
x = torch.tensor([3.0])

y = w * x + b 

loss = (y - 10.0) ** 2 

loss.backward()

print(w.grad)
print(b.grad)
print(loss)
# %%
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self , input_dim : int , hidden_dim : int , output_dim : int):

        super().__init__()

        self.first_layer = nn.Linear(input_dim , hidden_dim)
        self.second_layer = nn.Linear(hidden_dim , output_dim)

    def forward(self , x: torch.Tensor) -> torch.Tensor:

        x = F.relu(self.first_layer(x))
        logits = self.second_layer(x)

        return logits

def main():