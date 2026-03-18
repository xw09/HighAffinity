import torch
from torch import nn

from boltz.data import const
from boltz.model.layers.confidence_utils import compute_frame_pred, tm_function

def huber_loss(y, y_true, delta=0.5):
    """计算 Huber 损失"""
    # 计算误差\
    y_true_confirmed = torch.where(torch.isnan(y_true), y, y_true)
    error = y - y_true_confirmed
    # 判断误差的绝对值是否小于 δ
    is_small_error = torch.abs(error) <= delta
    # 使用 Huber 损失公式
    loss = torch.where(is_small_error, 0.5 * error**2, delta * (torch.abs(error) - 0.5 * delta))
    return loss.mean(), y_true

# Focal Loss的实现
def focal_loss(prob, targets, gamma=1, alpha=1):

    p_t = targets * prob + (1 - targets) * (1 - prob)

    focal_weight = torch.pow(1 - p_t, gamma)

    # 计算基础交叉熵
    bce_loss = -targets * torch.log(prob + 1e-8) - (1 - targets) * torch.log(1 - prob + 1e-8)
    
    loss = focal_weight * bce_loss
    
    # 平衡正负样本 
    loss = alpha * loss

    return loss.mean()

def affinity_loss(
    model_out,
    feats,
    multiplicity=1,
):
    # TODO no support for MD yet!
    # TODO only apply to the PDB structures not the distillation ones
    # breakpoint()
    affinity_pred_value_loss, rel_affinity_pred_value = huber_loss(
        model_out["affinity_pred_value"],
        # feats['record'][0].affinity.true_affinity
        torch.tensor([[v.affinity.true_affinity] if v.affinity.true_affinity else [torch.nan] for v in feats['record']], dtype=torch.float32).cuda()
    )

    # 使用 Focal Loss 计算二分类损失
    rel_affinity_probability_binary = torch.tensor([[v.affinity.true_class] for v in feats['record']], dtype=torch.float32).cuda()  # 真实标签
    λ_focal = 1.0  # 可以根据需要调整平衡系数，通常用来加重对正样本或负样本的关注。样本不平衡的情况，可以尝试调整 α 或 γ 来优化训练效果。
    affinity_probability_binary_loss = focal_loss(model_out["affinity_probability_binary"], rel_affinity_probability_binary, gamma=1, alpha=λ_focal)  

    # loss = (
    #     affinity_pred_value_loss*0.1
    #     + affinity_value_different_loss*0.9
    #     + affinity_probability_binary_loss
    # )
    loss = (affinity_pred_value_loss + affinity_probability_binary_loss)
    # print("affinity_pred_value_loss:",affinity_pred_value_loss)
    # print("affinity_probability_binary_loss:",affinity_probability_binary_loss)
    # print("loss:",loss)

    # dict_out = {
    #     "loss": loss,
    #     "loss_breakdown": {
    #         "affinity_pred_value_loss": affinity_pred_value_loss,
    #         "affinity_value_different_loss": affinity_value_different_loss,
    #         "affinity_probability_binary_loss": affinity_probability_binary_loss,
    #         "rel_affinity_pred_value": rel_affinity_pred_value,
    #         "rel_affinity_value_different": rel_affinity_value_different,
    #         "rel_affinity_probability_binary": rel_affinity_probability_binary,
    #     },
    # }
    dict_out = {
    "loss": loss,
    "loss_breakdown": {
        "affinity_pred_value_loss": affinity_pred_value_loss,
        "affinity_probability_binary_loss": affinity_probability_binary_loss,
        "rel_affinity_pred_value": rel_affinity_pred_value,
        "rel_affinity_probability_binary": rel_affinity_probability_binary,
    },
    }
    return dict_out

