import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from src.feature_builders.nuplan_feature_builder import NuplanFeatureBuilder

from .layers.common_layers import build_mlp
from .layers.transformer_encoder_layer import TransformerEncoderLayer
from .modules.agent_encoder import AgentEncoder
from .modules.map_encoder import MapEncoder
from .modules.trajectory_decoder import TrajectoryDecoder
import theseus as th
# no meaning, required by nuplan
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)
error_result=[]

class PlanningModel(TorchModuleWrapper):
    def __init__(
        self,
        dim=128,
        state_channel=6,
        polygon_channel=6,
        history_channel=9,
        history_steps=21,
        future_steps=80,
        encoder_depth=4,
        drop_path=0.2,
        num_heads=8,
        num_modes=6,
        use_ego_history=False,
        state_attn_encoder=True,
        state_dropout=0.75,
        feature_builder: NuplanFeatureBuilder = NuplanFeatureBuilder(),
        cost_weight=4
    ) -> None:
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )

        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps

        self.pos_emb = build_mlp(4, [dim] * 2)
        self.agent_encoder = AgentEncoder(
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            hist_steps=history_steps,
            drop_path=drop_path,
            use_ego_history=use_ego_history,
            state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout,
        )

        self.map_encoder = MapEncoder(
            dim=dim,
            polygon_channel=polygon_channel,
        )

        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )
        self.norm = nn.LayerNorm(dim)

        self.trajectory_decoder = TrajectoryDecoder(
            embed_dim=dim,
            num_modes=num_modes,
            future_steps=future_steps,
            out_channels=2,
        )
       
        self.agent_predictor = build_mlp(dim, [dim * 2, future_steps * 3], norm="ln")

        self.apply(self._init_weights)
        bs=32
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cost = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Linear(128, cost_weight), nn.Softmax(dim=-1))
        self.cost.to(device=device)
        self.dummy = torch.ones(bs, 1).to(device)
        self.register_buffer('constraint', torch.tensor([[10]]))
        self.constraint=self.constraint.expand(32,-1)

        #========================== dipp loss ============================#
        # device = torch.device("cuda:0")
        
        feature_len=cost_weight
         # define cost function
        cost_function_weights = [th.ScaleCostWeight(th.Variable(torch.rand(1), name=f'cost_function_weight_{i+1}')) for i in range(feature_len)]  
        # define control variable
        control_variables = th.Vector(dof=100, name="control_variables")
        # define prediction variable
        predictions = th.Variable(torch.empty(1, 32, 50, 3), name="predictions")
        # define ref_line_info
        ref_line_info = th.Variable(torch.empty(1, 1200, 5), name="ref_line_info")
        # define current state
        current_state = th.Variable(torch.empty(1, 33, 8), name="current_state")
        # set up objective
        objective = th.Objective()
        # print("init current_state:",current_state)
        self.objective = self.cost_function(objective, control_variables, current_state, predictions, ref_line_info, cost_function_weights)
        #return objective set

        
        self.prefix=""
        # set up optimizer
        if self.prefix=="val":
            self.optimizer = th.GaussNewton(objective, th.CholeskyDenseSolver, vectorize=False, max_iterations=50, step_size=0.2, abs_err_tolerance=1e-2)
        else:
            self.optimizer = th.GaussNewton(objective, th.LUDenseSolver, vectorize=False, max_iterations=2, step_size=0.4)
        self.layer = th.TheseusLayer(self.optimizer, vectorize=False)
        self.layer.to(device=device)
    def cost_function(self,objective, control_variables, current_state, predictions, ref_line, cost_function_weights, vectorize=True):
        timestep=50
        # comfort
        acc_cost = th.AutoDiffCostFunction([control_variables], self.acceleration, timestep, cost_function_weights[0], autograd_vectorize=vectorize, name="acceleration")
        objective.add(acc_cost)
        jerk_cost = th.AutoDiffCostFunction([control_variables], self.jerk, timestep-1, cost_function_weights[1], autograd_vectorize=vectorize, name="jerk")
        objective.add(jerk_cost)
        steering_cost = th.AutoDiffCostFunction([control_variables], self.steering, timestep, cost_function_weights[2], autograd_vectorize=vectorize, name="steering")
        objective.add(steering_cost)
        steering_change_cost = th.AutoDiffCostFunction([control_variables], self.steering_change, timestep-1, cost_function_weights[3], autograd_vectorize=vectorize, name="steering_change")
        objective.add(steering_change_cost)
        # safety
        # print("cost_function_weights[4]:",cost_function_weights[4].shape)
        # input()
        # safety_cost = th.AutoDiffCostFunction([control_variables], self.safety, 10, cost_function_weights[4], aux_vars=[predictions, current_state, ref_line], autograd_vectorize=vectorize, name="safety")
        # objective.add(safety_cost)
        # import csv
        # with open("/home/oem/zkf/planTF-dipp/error_result_csv.csv", mode='a') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(error_result)
        return objective
    def acceleration(self,optim_vars, aux_vars):
        timestep=50
        control = optim_vars[0].tensor.view(-1, timestep, 2)
        acc = control[:, :, 0]
        acc_row=torch.mean(acc,dim=1)
        result=torch.mean(acc_row)
        error_result.append(result)
        # print("acc.shape:",acc.shape) #[32, 50]
        return acc

    def jerk(self,optim_vars, aux_vars):
        timestep=50
        control = optim_vars[0].tensor.view(-1, timestep, 2)
        acc = control[:, :, 0]
        jerk = torch.diff(acc) / 0.1
        jerk_row=torch.mean(jerk,dim=1)
        result=torch.mean(jerk_row)
        error_result.append(result)
        # print("jerk.shape:",jerk.shape) #[32, 49]
        
        return jerk

    def steering(self,optim_vars, aux_vars):
        timestep=50
        control = optim_vars[0].tensor.view(-1, timestep, 2)
        steering = control[:, :, 1]
        steering_row=torch.mean(steering,dim=1)
        result=torch.mean(steering_row)
        error_result.append(result)
        return steering 

    def steering_change(self,optim_vars, aux_vars):
        timestep=50
        control = optim_vars[0].tensor.view(-1, timestep, 2)
        steering = control[:, :, 1]
        steering_change = torch.diff(steering) / 0.1
        steering_change_row=torch.mean(steering_change,dim=1)
        result=torch.mean(steering_change_row)
        error_result.append(result)
        return steering_change
    def safety(self,optim_vars, aux_vars):
        # aux_vars=[predictions, current_state,ref_line]
        timestep=50
        control = optim_vars[0].tensor.view(-1, timestep, 2)
        current_state = aux_vars[1].tensor#([batch, 11, 8]) 
        ego_current_state = current_state[:, 0]
        ego = self.bicycle_model(control, ego_current_state,"heading_v") #{batch，50，4} x y theta v
        neighbors = aux_vars[0].tensor.permute(0, 2, 1, 3) #torch.Size([bs, 50, 10, 3]) 1，10，50，3
        ego_len=current_state[:,0,-1]
        neighbors_len=current_state[:,1:,0]
        safe_error = []
        ls=[0, 2, 5, 9, 14, 20, 27, 35, 44, 54, 65, 77]
        ls_=[0, 2, 5, 9, 14, 19, 24, 29, 39, 49] 
        for t in ls_: # key frames
            distances = torch.norm(ego[:, t, :2].unsqueeze(1) - neighbors[:, t, :, :2], dim=-1).squeeze(1)
            distance, index = torch.min(distances, dim=1)
            s_eps = (ego_len + torch.index_select(neighbors_len, 1, index)[:, 0])/2 + 5
            error = (s_eps - distance) * (distance < s_eps)
            safe_error.append(error)
        safe_error = torch.stack(safe_error, dim=1) #[32,10] 32个agent 10帧的safe error
        safe_error_row_mean=torch.mean(safe_error,dim=1)
        result=torch.mean(safe_error_row_mean)
        error_result.append(result)
        # print("safe_error:",result) #代表这一批次的数据safe error平均值
        # print("safe_error.sahpe:",safe_error.shape)
        # input()
        # print("safe_error:",safe_error)
        return safe_error
    def project_to_frenet_frame(self,traj, ref_line):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        traj=traj.to(device)
        ref_line=ref_line.to(device)
        #计算traj上的点到ref_line上每一个点的欧氏距离x,y,每个traj的点占一行,列数为ref_line点数
        distance_to_ref = torch.cdist(traj[:, :, :2], ref_line[:, :, :2]).to(device) #{bs,num_traj_points,num_ref_points}
        #找出traj上每个点在ref_line上最近点的索引 k{bs,num_traj_points,3}
        k = torch.argmin(distance_to_ref, dim=-1).view(-1, traj.shape[1], 1).expand(-1, -1, 3).to(device)
        #找到这些index的ref_points
        ref_points = torch.gather(ref_line, 1, k).to(device)
        x_r, y_r, theta_r = ref_points[:, :, 0], ref_points[:, :, 1], ref_points[:, :, 2] 
        x, y = traj[:, :, 0].to(device), traj[:, :, 1].to(device)
        #相当于ref_line上的点就作为坐标尺度了,如果索引为1(代表traj点映射到了ref_line上第1个点),则s坐标轴下投影的距离就是0.1,
        # -200是一种缩放.比如把201的索引,缩放为索引1这样.
        s = 0.1 * (k[:, :, 0] - 200).to(device)#s表示沿着centerline走了多远
        #l=x^2+y^2 的平方根 并用sign判断是正向还是负向 ,l表示偏离参考线多少
        l = torch.sign((y-y_r)*torch.cos(theta_r)-(x-x_r)*torch.sin(theta_r)) * torch.sqrt(torch.square(x-x_r)+torch.square(y-y_r)).to(device)
        sl = torch.stack([s, l], dim=-1).to(device)
        return sl
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def bicycle_model(self,control, current_state,need):
        #control:a和steer (模型的预测量)
        dt = 0.1 # discrete time period [s]
        max_a = 5 # vehicle's accleration limits [m/s^2]
        max_d = 0.5 # vehicle's steering limits [rad]
        L = 3.089 # vehicle's wheelbase [m]
        
        x_0 = current_state[:, 0] # vehicle's x-coordinate [m]
        y_0 = current_state[:, 1] # vehicle's y-coordinate [m]
        theta_0 = current_state[:, 2] # vehicle's heading [rad]
        v_0 = current_state[:, 3] # vehicle's velocity [m/s]
        a = control[:, :, 0].clamp(-max_a, max_a) # vehicle's accleration [m/s^2]
        delta = control[:, :, 1].clamp(-max_d, max_d) # vehicle's steering [rad]

        # speed
        v = v_0.unsqueeze(1) + torch.cumsum(a * dt, dim=1)
        v = torch.clamp(v, min=0)
        
        # angle
        d_theta = v * delta / L # use delta to approximate tan(delta)
        theta = theta_0.unsqueeze(1) + torch.cumsum(d_theta * dt, dim=-1)
        theta = torch.fmod(theta, 2*torch.pi)
        cos_theta=torch.cos(theta)
        sin_theta=torch.sin(theta)
        
        # x and y coordniate
        x = x_0.unsqueeze(1) + torch.cumsum(v * torch.cos(theta) * dt, dim=-1)
        y = y_0.unsqueeze(1) + torch.cumsum(v * torch.sin(theta) * dt, dim=-1)
        if(need=="sincos"):
        # output trajectory
            traj = torch.stack([x, y, sin_theta, cos_theta], dim=-1)
        elif(need=="heading_v"):
            traj = torch.stack([x, y, theta, v], dim=-1)
        elif(need=="heading"):
            traj = torch.stack([x, y, theta], dim=-1)
        return traj
    def forward(self, data):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]
        current_state=data["current_state"]
        bs, A = agent_pos.shape[0:2] #A 有多少个agent
        # print("A",A)
        
        
        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
        pos = torch.cat(
            [position, torch.stack([angle.cos(), angle.sin()], dim=-1)], dim=-1
        )
        pos_embed = self.pos_emb(pos)

        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)

        x = torch.cat([x_agent, x_polygon], dim=1) + pos_embed

        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm(x) #预测了所有人的轨迹。 #[32, 139, 128] 中间这个不一定，有多少个agent就有多少
        # print("x:",x.shape)
        trajectory, probability = self.trajectory_decoder(x[:, 0])#第0个是ego
        # print("trajectory in trajectory_decoder:",trajectory.shape)
        # input()
        trajectory=trajectory.to(device)
        probability=probability.to(device)
        # add biycle model : traj (2 dim) to traj (4)
        # trajectory = self.bicycle_model(trajectory, current_state,"sincos")
        # test=self.agent_predictor(x[:, 1:A]) #[32, 32, 160]
        # print("test:",test.shape)
        # input()
        # self.agent_predictor = build_mlp(dim, [dim * 2, future_steps * 2], norm="ln")
        prediction = self.agent_predictor(x[:, 1:A]).view(bs, -1, self.future_steps, 3).to(device) #从1往后是其他的agent轨迹
        # prediction[32, 32，80, 3]
        # print("device_dummy:",self.dummy.device)
        self.cost=self.cost.to(self.dummy.device)
        # print("self.cost:",self.cost.device)
        # cost_function_weights = self.cost(self.dummy)#bs,5
        # print("cost_function_weights ol:",cost_function_weights.shape)
        # cost_function_weights = torch.cat([self.cost(self.dummy)[:,:4], self.constraint], dim=-1)
        cost_function_weights = self.cost(self.dummy)[:,:4]
        # print("cost_function_weights:",cost_function_weights.shape)
        # input()
        
        out = {
            "trajectory": trajectory,
            "probability": probability,
            "prediction": prediction,
            "weight":cost_function_weights
        }

        if not self.training:
            best_mode = probability.argmax(dim=-1)
            output_trajectory = trajectory[torch.arange(bs), best_mode]
            output_trajectory=output_trajectory.to(device=current_state.device)
            out["output_trajectory"]  = self.bicycle_model(output_trajectory, current_state,"heading")
            # angle = torch.atan2(output_trajectory[..., 3], output_trajectory[..., 2])
            # out["output_trajectory"] = torch.cat(
            #     [output_trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
            # )
        return out
