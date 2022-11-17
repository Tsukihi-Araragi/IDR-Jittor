from jittor.utils.pytorch_converter import convert
pytorch_code="""
x.requires_grad_(True)
y = self.forward(x)[:,:1]
d_output = torch.ones_like(y, requires_grad=False, device=y.device)
gradients = torch.autograd.grad(
    outputs=y,
    inputs=x,
    grad_outputs=d_output,
    create_graph=True,
    retain_graph=True,
    only_inputs=True)[0]
return gradients.unsqueeze(1)
"""

jittor_code = convert(pytorch_code)
print(jittor_code)