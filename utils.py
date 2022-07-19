def clonumpy(tensor):
    return tensor.clone().detach().numpy().flatten()

