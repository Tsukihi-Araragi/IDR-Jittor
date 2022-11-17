import jittor as jt
from jittor import Module
from jittor import nn
from jittor import init
import numpy as np

if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/d/pancheng/Project/IDR-Jittor/code')
    
from model.embedder import get_embedder
class ImplicitNetwork(Module):
    def __init__(self, 
                 feature_vector_size,
                 d_in,
                 d_out,
                 dims,
                 geometric_init=True,
                 bias=1.0,
                 skip_in=(),
                 multires=0):
        super().__init__()
        dims = [d_in] + dims + [d_out + feature_vector_size]
        print("dims[-1]:    ", dims[-1])
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        
        
        # jt.init._no_grad_trunc_normal_
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
                
            lin = nn.Linear(dims[l], out_dim)
            # print("lin.weight.size(): ", lin.weight.size())
            # if geometric_init:
            #     if l == self.num_layers - 2:
            #         init.gauss_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
            #         init.constant_(lin.bias, -bias)
            #     elif multires > 0 and l == 0:
            #         init.constant_(lin.bias, 0.0)
            #         init.constant_(lin.weight[:, 3:], 0.0)
            #         init.gauss_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            #     elif multires > 0 and l in self.skip_in:
            #         init.constant_(lin.bias, 0.0)
            #         init.gauss_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            #         init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
            #     else:
            #         init.constant_(lin.bias, 0.0)
            #         init.gauss_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    

            setattr(self, "lin" + str(l), lin)
        self.softplus = nn.Softplus(beta=100)
    
    
    def execute(self, input, compute_grad=False):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input
        # print("input: ", input)
        for l in range(0, self.num_layers - 1):
            # print("lllllllllllllll: ", l, self.num_layers)
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = jt.contrib.concat([x, input], dim=1) / np.sqrt(2)
                
            # print("88888888888888888888:  ", x.shape)

            x = lin(x)
            # print("99999999999999999999：  ", x.shape)
            if l < self.num_layers - 2:
                x = self.softplus(x)
            # print("size  max min:   ", x.size(), jt.max(jt.max(x, 1)[0]), jt.min(jt.min(x, 1)[0]))
            # print(x[0][:10])
        return x
    
    def gradient(self, x):
        x.start_grad()
        y = self.execute(x)[:, :1]
        # d_output = jt.ones_like(y, requires_grad=False, device=y.device)
        # gradients = jt.autograd.grad(outputs=y, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True)[0]
        return jt.grad(y, x)

class RenderingNetwork(Module):
    def __init__(self,
                 feature_vector_size,
                 mode,
                 d_in,
                 d_out,
                 dims,
                 multires_view=0):
        
        super().__init__()
        
        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]
        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)
        
        self.num_layers = len(dims)
        
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            setattr(self, "lin" + str(l), lin)
        
        self.relu = nn.Relu()
        self.tanh = nn.Tanh()
        
    def execute(self, points, normals, view_dirs, feature_vectors):
        if self.embed_fn is not None:
            view_dirs = self.embed_fn(view_dirs)
            
        if self.mode == 'idr':
            rendering_input = jt.contrib.concat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = jt.contrib.concat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = jt.contrib.concat([points, view_dirs, feature_vectors], dim=-1)
            
        x = rendering_input
        # print("input: ", input)
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = (jt.contrib.concat([x, input], dim=1) / np.sqrt(2))
                
            # print("88888888888888888888:  ", x.shape)

            x = lin(x)
            # print("99999999999999999999：  ", x.shape)
            if l < self.num_layers - 2:
                x = self.softplus(x)
                
        return x
        
class IDRNetwork(Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))



model = ImplicitNetwork(256, 3, 1, [ 512, 512, 512, 512, 512, 512, 512, 512 ], True, 0.6, [4], 6)
input = jt.rand(1, 3)
input = jt.array((1., 2., 3., 1., 2., 3.)).reshape(2, 3)
print(input.size())
print("input, input: ", input)

# input[0] = -0.4182
# input[1] = -0.9914
# input[2] = -0.3587
# # input = jt.Var(-0.4182, -0.9914, -0.3587)
# input[0] = -input[0]
print("input: ", input)
output = model(input)
print("output.size(): ", output.size())

print(jt.max(output, 1), jt.min(input, 1))
print(jt.grad(output[:, :1], input))
print(model.gradient(input))


