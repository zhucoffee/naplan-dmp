import logging
import os
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import MetricCollection

from src.metrics import MR, minADE, minFDE
from src.optim.warmup_cos_lr import WarmupCosLR

logger = logging.getLogger(__name__)


class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model: TorchModuleWrapper,
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.objective = self.model.objective

    def on_fit_start(self) -> None:
        metrics_collection = MetricCollection(
            {
                "minADE1": minADE(k=1).to(self.device),
                "minADE6": minADE(k=6).to(self.device),
                "minFDE1": minFDE(k=1).to(self.device),
                "minFDE6": minFDE(k=6).to(self.device),
                "MR": MR().to(self.device),
            }
        )
        self.metrics = {
            "train": metrics_collection.clone(prefix="train/"),
            "val": metrics_collection.clone(prefix="val/"),
        }
    def bicycle_model_formuti(self,control_, current_state,need):
        #control:a和steer (模型的预测量)
        dt = 0.1 # discrete time period [s]
        max_a = 5 # vehicle's accleration limits [m/s^2]
        max_d = 0.5 # vehicle's steering limits [rad]
        L = 3.089 # vehicle's wheelbase [m]
        
        x_0 = current_state[:, 0] # vehicle's x-coordinate [m]
        y_0 = current_state[:, 1] # vehicle's y-coordinate [m]
        theta_0 = current_state[:, 2] # vehicle's heading [rad]
        v_0 = current_state[:, 3] # vehicle's velocity [m/s]

        processed_trajectory = torch.zeros(control_.shape[0],control_.shape[1],control_.shape[2],4)
        # print("processed_trajectory:",processed_trajectory.shape)

        for modality in range(control_.size(1)):
            modality_trajectory = control_[:, modality, :, :]
            control=modality_trajectory
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
            if(need=="cossin"):
            # output trajectory
                traj = torch.stack([x, y, cos_theta, sin_theta], dim=-1)
            elif(need=="heading_v"):
                traj = torch.stack([x, y, theta, v], dim=-1)
            elif(need=="heading"):
                traj = torch.stack([x, y, theta], dim=-1)
            t=processed_trajectory[:, modality, :, :]
            # print("traj:",traj.shape)
            # print("t:",t.shape)
            # input()
            processed_trajectory[:, modality, :, :] = traj

        return processed_trajectory
    def bicycle_model(self,control, current_state,need):
        # print("control:",control.shape)
        # input()
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
        if(need=="cossin"):
        # output trajectory
            traj = torch.stack([x, y, cos_theta, sin_theta], dim=-1)
        elif(need=="heading_v"):
            traj = torch.stack([x, y, theta, v], dim=-1)
        elif(need=="heading"):
            traj = torch.stack([x, y, theta], dim=-1)
        return traj
    def _step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str
    ) -> torch.Tensor:
        features, _, _ = batch
        # print("device_feature:",features["feature"].data.device)
        res = self.forward(features["feature"].data)
        self.model.prefix=prefix
        losses = self._compute_objectives(res, features["feature"].data)
        metrics = self._compute_metrics(res, features["feature"].data, prefix)
        self._log_step(losses["loss"], losses, metrics, prefix)
        # if(prefix=="train"):
        #   import csv
        #   with open("/home/oem/zkf/planTF-dipp/loss_train_csv_0111_nosafe.csv", mode='a') as file:
        #         writer = csv.writer(file)
        #         writer.writerow([x.item() for x in losses.values()])
        #         # writer.writerow(losses.values().item())  # 写入一行数据到 CSV 文件
        return losses["loss"]

    def _compute_objectives(self, res, data) -> Dict[str, torch.Tensor]:
        # res=res.to(data.device)
        current_state=data["current_state"]
        trajectory, probability, prediction = (
            res["trajectory"],#[32, 6, 80, 2]
            res["probability"], #[32, 6]
            res["prediction"],#[32, 32, 80, 3]
        )
#====================================================20240122 post-optimizer=========================================#
        targets = data["agent"]["target"]
        # print("targets[:, 1:]:",targets[:, 1:].shape) #[32, 32, 80, 3]
        # input()
        current_state=torch.zeros(data["current_state"].shape[0],33,8)
        current_state[:, 0,:7]=data["current_state"]
        print("ego_cur_state_shape:",data["current_state"].shape)
        print("ego_cur_state:",data["current_state"])
        input()
        # ======= 是否需要处理为speed方向的v，dipp中没有处理，使用的是velo_x==========#
        # ego_vx=current_state[:,0,3]
        # ego_heading=current_state[:,0,2] #rad
        # current_state[:,0,3]=ego_vx/torch.cos(ego_heading)
        # ======= 是否需要处理为speed方向的v，dipp中没有处理，使用的是velo_x==========#

        current_state[:, 0,7]=data["agent"]["shape"][:,0,0,1]#ego_len
        current_state[:, 1:,0]=data["agent"]["shape"][:,1:,0,1]#neighbor的len
        ego_current_state=current_state[:,0] #[bs,7] 自行车模型里只需要前四个是 x y heading v

        transformed_trajectory = torch.zeros((32, 6, 80, 3))
        for i in range(trajectory.shape[0]):
            one_mode_trajectory=trajectory[:,i,:,:]
            transformed_trajectory[:,i,:,:]=self.bicycle_model(one_mode_trajectory,ego_current_state,"heading")
        ego_target_pos, ego_target_heading = targets[:, 0, :, :2], targets[:, 0, :, 2]
        ego_target = torch.cat(
            [
                ego_target_pos,
                torch.stack(
                    [ego_target_heading.cos(), ego_target_heading.sin()], dim=-1
                ),
            ],
            dim=-1,
        )
        ego_target_heading_=ego_target_heading.unsqueeze(-1)
        ego_target_xyheading=torch.cat([ego_target_pos,ego_target_heading_],dim=-1).to(device)
        ade = torch.norm(transformed_trajectory[..., :2] - ego_target[:, None, :, :2], dim=-1)
        best_mode = torch.argmin(ade.sum(-1), dim=-1)
        best_traj = trajectory[torch.arange(trajectory.shape[0]), best_mode] # dim=2,for dipp control
        transformed_best_traj = transformed_trajectory[torch.arange(transformed_trajectory.shape[0]), best_mode] #dim=4,for l1 loss
        output=best_traj


        planner_inputs = {
                "control_variables": output.view(-1, 160), # initial control sequence
                "predictions": prediction, # prediction for surrounding vehicles 
                "ref_line_info": ref_line_info, #navigation
                "current_state": current_state #ego and agent
        }
    
        # plan = self.bicycle_model(plan, ego_current_state,"heading").to(device)
        
        agent_reg_loss = F.smooth_l1_loss(
            prediction[agent_mask], agent_target[agent_mask][:, :2]
        )
        
        # ego_reg_loss = F.smooth_l1_loss(best_traj_, ego_target)
        prediction_loss=agent_reg_loss



        # # # result:
        # # loss=0.5*prediction_loss+score_loss+plan_loss+1e-3*plan_cost
        # # dict_loss={
        # #     "loss": loss,
        # #     "prediction_loss":prediction_loss, #计算的是除了自车以外的prediction loss
        # #     "score_loss":score_loss, 
        # #     "plan_loss":plan_loss,
        # #     "plan_cost":plan_cost,
        # # }
        # # return dict_loss
#====================================================20240122 post-optimizer=========================================#
        targets = data["agent"]["target"]
        valid_mask = data["agent"]["valid_mask"][:, :, -trajectory.shape[-2] :]
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        current_state=current_state.to(device)
        trajectory=trajectory.to(device)
        probability=probability.to(device)
        prediction=prediction.to(device)
        targets=targets.to(device)
        valid_mask=valid_mask.to(device)
        cost_function_weights=res["weight"].to(device=device)

        #best traj
        trajectory_=self.bicycle_model_formuti(trajectory, current_state,"cossin") # x y con sin
        # print("trajectory:",trajectory.shape)
        targets=targets.to(trajectory_.device)
        # valid_mask=valid_mask.to(trajectory.device)

        ego_target_pos, ego_target_heading = targets[:, 0, :, :2], targets[:, 0, :, 2]
        ego_target = torch.cat(
            [
                ego_target_pos,
                torch.stack(
                    [ego_target_heading.cos(), ego_target_heading.sin()], dim=-1
                ),
            ],
            dim=-1,
        )
        # print(ego_target_pos.shape)
        # print("ego_target_heading",ego_target_heading.shape)
        # input()
        ego_target_heading_=ego_target_heading.unsqueeze(-1)
        ego_target_xyheading=torch.cat([ego_target_pos,ego_target_heading_],dim=-1).to(device)
        ade = torch.norm(trajectory_[..., :2] - ego_target[:, None, :, :2], dim=-1)
        best_mode = torch.argmin(ade.sum(-1), dim=-1)
        best_traj = trajectory[torch.arange(trajectory.shape[0]), best_mode] # dim=2,for dipp control
        best_traj_ = trajectory_[torch.arange(trajectory.shape[0]), best_mode] #dim=4,for l1 loss
        best_traj_=best_traj_.to(ego_target.device)
        # print("best_traj:",best_traj.shape)
        # print("best_traj_:",best_traj_.shape)
        # input()

        #cost function 
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        batch_size=32
        output=best_traj.to(device=device)
        output=output[:,:50,:]
        # print("output.shape:",output.shape)
        # input()
        prediction_=res["prediction"][:,:,:50,:].to(device=device)  # torch.Size([32, 32, 50, 2])
        # print("pre:",prediction.shape)
        zeros = torch.zeros(32,32, 50, 1).to(device)
        prediction_=torch.cat((prediction_, zeros), dim=3)
        # prediction=prediction.unsqueeze(0)
        cost_function_weights=res["weight"].to(device=device)
        # cost_function_weights[:,-1]=10
        ref_line_info=torch.randn(batch_size,1200,5).to(device=device)
        current_state=torch.zeros(32,33,8).to(device=device)
        current_state[:, 0,:7]=data["current_state"]
        current_state[:, 0,7]=data["agent"]["shape"][:,0,0,1]#ego_len
        current_state[:, 1:,0]=data["agent"]["shape"][:,1:,0,1]#neighbor的len

        planner_inputs = {
                "control_variables": output.view(-1, 100), # initial control sequence
                "predictions": prediction_, # prediction for surrounding vehicles 
                "ref_line_info": ref_line_info, #navigation
                "current_state": current_state #ego
        }
        for i in range(cost_function_weights.shape[1]):
            planner_inputs[f'cost_function_weight_{i+1}'] = cost_function_weights[:, i].unsqueeze(1)
        
        
        # print("model.prefix:",self.model.prefix)
        final_values, info = self.model.layer.forward(planner_inputs)
        plan = final_values["control_variables"].view(-1, 50, 2).to(device) #([32, 50, 2])
        ego_current_state = current_state[:, 0].to(device)
        # print("plan:",plan.shape)
        # print('current_state:',current_state.shape) #e([32, 33, 8])
        plan = self.bicycle_model(plan, ego_current_state,"heading").to(device)

        # plan_cost=self.model.objective(planner_inputs["control_variables"],planner_inputs['current_state'],planner_inputs['predictions'],planner_inputs['ref_lin+e_info'],cost_function_weights)
        plan_cost=self.objective.error_metric().mean() / self.objective.dim()

        # loss orin 
        agent_target, agent_mask = targets[:, 1:], valid_mask[:, 1:]
        print("targets[:, 1:]:",targets[:, 1:].shape)
        input()
        agent_target=agent_target.to(device)
        agent_mask=agent_mask.to(device)

        ego_reg_loss = F.smooth_l1_loss(best_traj_, ego_target)
        probability=probability.to(best_mode.device)
        ego_cls_loss = F.cross_entropy(probability, best_mode.detach())
        # print("prediction.device:",prediction.device)
        prediction=prediction.to(device=agent_target.device)
        # print("prediction.device:",prediction.device)
        # input()
        agent_reg_loss = F.smooth_l1_loss(
            prediction[agent_mask], agent_target[agent_mask][:, :2]
        )

        loss_orin = 0.5*ego_reg_loss + ego_cls_loss + 0.5*agent_reg_loss
        prediction_loss=ego_reg_loss+agent_reg_loss
        score_loss=ego_cls_loss

        # plan loss
        # ego_target_xyheading=
        plan_loss=F.smooth_l1_loss(plan,ego_target_xyheading[:,:50,:])
        plan_loss+=F.smooth_l1_loss(plan[:, -1], ego_target_xyheading[:,-1, :])

        # result:
        loss=0.5*prediction_loss+score_loss+plan_loss+1e-3*plan_cost
        dict_loss={
            "loss": loss,
            "prediction_loss":prediction_loss,
            "score_loss":score_loss,
            "plan_loss":plan_loss,
            "plan_cost":plan_cost,
        }
        return dict_loss

    def _compute_metrics(self, output, data, prefix) -> Dict[str, torch.Tensor]:
        # data=data.to(output.device)
        metrics = self.metrics[prefix](output, data["agent"]["target"][:, 0])
        return metrics

    def _log_step(
        self,
        loss: torch.Tensor,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = "loss",
    ) -> None:
        self.log(
            f"loss/{prefix}_{loss_name}",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        for key, value in objectives.items():
            self.log(
                f"objectives/{prefix}_{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        if metrics is not None:
            self.log_dict(
                metrics,
                prog_bar=(prefix == "val"),
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )

    def training_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "train")

    def validation_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "val")

    def test_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "test")

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Propagates a batch of features through the model.

        :param features: features batch
        :return: model's predictions
        """
        return self.model(features)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        # Get optimizer
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )

        # Get lr_scheduler
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        return [optimizer], [scheduler]
