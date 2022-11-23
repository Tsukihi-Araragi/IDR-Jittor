if __name__ == "__main__":
    import sys
    sys.path.append('/mnt/d/pancheng/Project/IDR-Jittor/code')
import jittor as jt
from jittor import Module
from jittor import nn
from jittor import init
import numpy as np
from dataset import scene_dataset
from utils import rend_util
from ray_tracing import RayTracing
from sample_network import SampleNetwork
from model.embedder import get_embedder

jt.flags.use_cuda = True
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
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')
        
    def execute(self):
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape(-1)
        
        self.implicit_network.eval()
        
        with jt.no_grad():
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                 cam_loc=cam_loc,
                                                                 object_mask=object_mask,
                                                                 ray_directions=ray_dirs)
        
        self.implicit_network.train()
        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)

        sdf_output = self.implicit_network(points)[:, 0:1]
        ray_dirs = ray_dirs.reshape(-1, 3)
        
        if self.training:
            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask]
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]
            N = surface_points.shape[0]
        
            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = jt.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = jt.cat([eikonal_points, eikonal_pixel_points], 0)

            points_all = jt.cat([surface_points, eikonal_points], dim=0)

            output = self.implicit_network(surface_points)
            surface_sdf_values = output[:N, 0:1].detach()

            g = self.implicit_network.gradient(points_all)
            surface_points_grad = g[:N, 0, :].clone().detach()
            grad_theta = g[N:, 0, :]

            differentiable_surface_points = self.sample_network(surface_output,
                                                                surface_sdf_values,
                                                                surface_points_grad,
                                                                surface_dists,
                                                                surface_cam_loc,
                                                                surface_ray_dirs)

        else:
            surface_mask = network_object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None
        # dataloader = scene_dataset.SceneDataset(False, 'DTU', [1200, 1600], 65, batch_size=2)
        # for data_index, (idx, model_input, ground_truth) in enumerate(dataloader):
        #     print("data_index: ", data_index)
        #     print("idx: ", idx)
        #     print("object_mask: ", model_input['object_mask'].size())
        #     print("uv: ", model_input['uv'].size())
        #     print("intrinsics: ", model_input['intrinsics'].size())
        #     print("pose: ", model_input['pose'].size())
            
        #     ray_dirs, cam_loc = rend_util.get_camera_params(model_input['uv'], model_input['pose'], model_input['intrinsics'])
        #     print("ray_dirs: ", ray_dirs.size())
        #     print("cam_loc: ", cam_loc.size())
        #     print(cam_loc, ray_dirs[0][0])
        


# model = ImplicitNetwork(256, 3, 1, [ 512, 512, 512, 512, 512, 512, 512, 512 ], True, 0.6, [4], 6)
# input = jt.rand(1, 3)
# input = jt.array((1., 2., 3., 1., 2., 3.)).reshape(2, 3)
# print(input.size())
# print("input, input: ", input)

# # input[0] = -0.4182
# # input[1] = -0.9914
# # input[2] = -0.3587
# # # input = jt.Var(-0.4182, -0.9914, -0.3587)
# # input[0] = -input[0]
# print("input: ", input)
# output = model(input)
# print("output.size(): ", output.size())

# print(jt.max(output, 1), jt.min(input, 1))
# print(jt.grad(output[:, :1], input))
# print(model.gradient(input))


model = IDRNetwork(None)

model.execute()