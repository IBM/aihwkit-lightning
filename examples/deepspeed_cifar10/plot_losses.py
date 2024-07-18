import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":

    triton_losses_fp16_grad = torch.load("losses_cifar10_triton_True.th")
    torch_losses = torch.load("losses_cifar10_triton_False.th")

    plt.figure(figsize=(7, 5))
    plt.plot(triton_losses_fp16_grad, label="Triton FP-16 grad")
    plt.plot(torch_losses, label="Torch")
    plt.legend()
    plt.savefig("loss_comp.pdf")
