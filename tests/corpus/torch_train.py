"""A small PyTorch image classifier training loop."""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

images = torch.randn(64, 3, 32, 32)
targets = torch.randint(0, 10, (64,))
loader = DataLoader(TensorDataset(images, targets), batch_size=8, shuffle=True)  # expect: ML.PYTORCH.RANDOMNESS_CONTROL_CHECKER_PYTORCH_DATALOADER
val_loader = DataLoader(TensorDataset(images, targets), batch_size=8)


class Net(nn.Module):  # expect: ML.MISSING_DOCSTRING
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)
        self.head = nn.Linear(8 * 30 * 30, 10)  # expect: ML.MAGIC_NUMBER, ML.MAGIC_NUMBER, ML.MAGIC_NUMBER

    def forward(self, x):
        return self.head(torch.flatten(self.conv(x), 1))


class Wrapper(Net):  # expect: ML.MISSING_DOCSTRING
    def forward(self, x):
        return super().forward(x)


model = Wrapper()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(20):
    for batch, labels in loader:
        loss = loss_fn(model(batch), labels)
        loss.backward()  # expect: ML.PYTORCH.GRADIENT_CLEAR_CHECKER
        optimizer.step()

for batch, labels in val_loader:
    logits = model.forward(batch)  # expect: ML.PYTORCH.NET_FORWARD_CHECKER
    probs = torch.log_softmax(logits, dim=1)

# File-level findings (reported at line 0):
# expect-file: ML.PYTORCH.BATCH_NORMALISATION_CHECKER
# expect-file: ML.PYTORCH.DATA_AUGMENTATION_CHECKER
# expect-file: ML.PYTORCH.DETERMINISTIC_ALGORITHM_USAGE_CHECKER
# expect-file: ML.PYTORCH.LEARNING_RATE_SCHEDULER_CHECKER
# expect-file: ML.PYTORCH.LOGGING_CHECKER
# expect-file: ML.PYTORCH.MODEL_EVALUATION_CHECKER
# expect-file: ML.PYTORCH.RANDOMNESS_CONTROL_CHECKER
