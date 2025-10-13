import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    def __init__(self, input_dim: int = 10, hidden_dim: int = 60, output_dim: int = 2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)


def init_model_with_fixed_random_weights(model: nn.Module):
    with torch.no_grad():
        model.fc1.weight.copy_(torch.randn_like(model.fc1.weight))
        model.fc1.bias.copy_(torch.randn_like(model.fc1.bias))
        model.fc2.weight.copy_(torch.randn_like(model.fc2.weight))
        model.fc2.bias.copy_(torch.tensor([0.2, 0.01]))

def test_fcn_with_random_weights(save_weights: bool):
    batch_size = 3
    model = FCN()
    init_model_with_fixed_random_weights(model)
    inputs = torch.randn(batch_size, 10)
    outputs = model(inputs)

    for i in range(batch_size):
        print("batch: {}, output: {}".format(i, outputs[i].detach().cpu()))
    
    if save_weights:
        ckpt_path = "../ckpts/fcn_weights.pt"
        torch.save(model.state_dict(), ckpt_path)
        print("FCN weights have been saved to {}".format(ckpt_path))

def load_and_inference(path: str):
    print("loading FCN weights from {}".format(path))
    model = FCN()
    model.load_state_dict(torch.load(path, map_location="cpu"))

    batch_size = 3
    inputs = torch.randn(batch_size, 10)
    outputs = model(inputs)

    for i in range(batch_size):
        print("batch: {}, output: {}".format(i, outputs[i].detach().cpu()))

if __name__ == "__main__":
    torch.manual_seed(1)
    # test_fcn_with_random_weights(True)
    load_and_inference("../ckpts/fcn_weights.pt")
