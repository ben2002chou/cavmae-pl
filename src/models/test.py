import torch

a = torch.tensor([True, False, True, True])
b = torch.tensor([False, False, False, False])

if a is not None and b is not None:
    print("A and B")
elif a is not None and b is None:
    print("A")
elif b is not None and a is None:
    print("B")
else:
    print(None)
