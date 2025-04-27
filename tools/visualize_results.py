#!/usr/bin/env python
"""
BEVFormer检测结果可视化工具

该脚本用于可视化BEVFormer模型在多视角相机输入下的3D检测结果。
它支持将检测结果以多视图方式显示，并在图像上绘制检测到的3D边界框。
"""
import os
import sys
import os.path as osp

# 检查是否在正确的conda环境中
if 'bevformer' not in sys.executable:
    print("Please run this script in the bevformer conda environment!")
    print("Use: conda activate bevformer")
    sys.exit(1)

import argparse
import mmcv
import torch
import cv2
import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
import copy
from mmcv.parallel.data_container import DataContainer
import matplotlib.pyplot as plt
import tempfile
import shutil

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmdet3d.apis import single_gpu_test, init_model, inference_detector
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.core.visualizer import show_multi_modality_result, show_result
from mmdet3d.core.bbox import Box3DMode, get_box_type
from mmdet3d.core.bbox.structures import (
    LiDARInstance3DBoxes, 
    CameraInstance3DBoxes, 
    DepthInstance3DBoxes
)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Visualize BEVFormer detection results')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument(
        '--out-dir', 
        default='vis_results', 
        help='output directory for visualization results'
    )
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='show debugging information'
    )
    return parser.parse_args()

def project_3d_box_to_2d(corners_3d, projection_matrix):
    """
    将3D边界框角点投影到2D图像平面
    
    Args:
        corners_3d (np.ndarray): 3D边界框的8个角点坐标
        projection_matrix (np.ndarray): 投影矩阵
        
    Returns:
        np.ndarray | None: 2D投影点数组，如果任何点在相机后方则返回None
    """
    corners_2d = []
    for corner in corners_3d:
        # 转换为齐次坐标
        corner_homogeneous = np.append(corner, 1)
        # 投影到图像空间
        corner_2d = np.dot(projection_matrix, corner_homogeneous)
        # 如果点在相机后方则跳过
        if corner_2d[2] <= 0:
            return None
        # 执行齐次除法
        corner_2d = corner_2d[:2] / corner_2d[2]
        corners_2d.append(corner_2d)
    
    return np.array(corners_2d)

def draw_3d_box(img, corners_2d, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制3D边界框
    
    Args:
        img (np.ndarray): 图像数组
        corners_2d (np.ndarray): 投影到2D的边界框角点
        color (tuple): BGR颜色
        thickness (int): 线条粗细
    """
    corners_2d = corners_2d.astype(np.int32)
    
    try:
        # 确保图像和坐标有效
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)
            
        # 绘制底面（前4个角点）
        for k in range(4):
            cv2.line(
                img, 
                tuple(corners_2d[k]), 
                tuple(corners_2d[(k+1)%4]), 
                color=color, 
                thickness=thickness
            )
        
        # 绘制顶面（后4个角点）
        for k in range(4, 8):
            cv2.line(
                img, 
                tuple(corners_2d[k]), 
                tuple(corners_2d[4 + (k-4+1)%4]), 
                color=color, 
                thickness=thickness
            )
        
        # 绘制连接顶面和底面的线条
        for k in range(4):
            cv2.line(
                img, 
                tuple(corners_2d[k]), 
                tuple(corners_2d[k+4]), 
                color=color, 
                thickness=thickness
            )
    except Exception as e:
        print(f"Error in draw_3d_box: {e}")

def get_projection_matrix(img_metas, view_idx):
    """
    获取特定视图的投影矩阵
    
    Args:
        img_metas: 图像元数据
        view_idx (int): 相机视图索引
        
    Returns:
        np.ndarray | None: 投影矩阵，如果找不到则返回None
    """
    lidar2img = None
    
    # 尝试不同的数据结构获取投影矩阵
    # 情况1: img_metas是字典列表，每个视图一个字典
    if isinstance(img_metas, list):
        if view_idx < len(img_metas) and isinstance(img_metas[view_idx], dict) and 'lidar2img' in img_metas[view_idx]:
            lidar2img = img_metas[view_idx]['lidar2img']
    
    # 情况2: img_metas是单个字典，包含所有视图的数据
    elif isinstance(img_metas, dict) and 'lidar2img' in img_metas:
        if isinstance(img_metas['lidar2img'], list) and view_idx < len(img_metas['lidar2img']):
            lidar2img = img_metas['lidar2img'][view_idx]
    
    # 情况3: img_metas是字典列表，但投影矩阵存储在第一个字典中
    if lidar2img is None and isinstance(img_metas, list) and len(img_metas) > 0:
        first_meta = img_metas[0]
        if isinstance(first_meta, dict) and 'lidar2img' in first_meta:
            if isinstance(first_meta['lidar2img'], list) and view_idx < len(first_meta['lidar2img']):
                lidar2img = first_meta['lidar2img'][view_idx]
    
    # 转换为numpy数组
    if lidar2img is not None:
        if isinstance(lidar2img, torch.Tensor):
            lidar2img = lidar2img.cpu().numpy()
        elif isinstance(lidar2img, list):
            # 处理嵌套列表结构
            if isinstance(lidar2img[0], list) or isinstance(lidar2img[0], np.ndarray):
                lidar2img = np.array(lidar2img[0])
            else:
                lidar2img = np.array(lidar2img)
    
    return lidar2img

def denormalize_image(img_tensor, img_norm_cfg=None):
    """
    将归一化的图像张量转换回原始图像
    
    Args:
        img_tensor (torch.Tensor): 归一化的图像张量
        img_norm_cfg (dict): 图像归一化配置
        
    Returns:
        np.ndarray: 反归一化的图像数组，范围在[0, 255]，uint8类型
    """
    # 设置默认归一化参数
    if img_norm_cfg:
        mean = torch.tensor(img_norm_cfg['mean']).view(-1, 1, 1)
        std = torch.tensor(img_norm_cfg['std']).view(-1, 1, 1)
        to_rgb = img_norm_cfg.get('to_rgb', True)
    else:
        # 使用默认值
        mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        to_rgb = True
    
    # 反归一化
    img_np = (img_tensor * std + mean).numpy()
    
    # 确保图像有正确的形状用于转置
    if len(img_np.shape) == 3 and img_np.shape[0] == 3:  # [C, H, W]
        img_np = img_np.transpose(1, 2, 0)
    
    # 如果需要，将BGR转换为RGB
    if not to_rgb:  # 如果to_rgb为False，表示图像是BGR格式
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    
    # 确保图像是连续内存布局，避免OpenCV错误
    img_np = np.ascontiguousarray(np.clip(img_np, 0, 255).astype(np.uint8))
    return img_np

def visualize_multi_modality_result(data, result, out_dir, fig, axes, 
                                   score_thr=0.3, debug=False, batch_idx=0, show=True):
    """
    多模态数据的自定义可视化函数
    
    Args:
        data: 输入数据
        result: 检测结果
        out_dir: 输出目录
        fig: matplotlib图形对象
        axes: matplotlib子图对象列表
        score_thr: 得分阈值
        debug: 是否显示调试信息
        batch_idx: 批次索引
        show: 是否显示图像
        
    Returns:
        bool: 如果用户选择退出，则返回True
    """
    # 提取图像和元数据
    if debug:
        print("Data keys:", data.keys())
    
    try:
        # 提取图像
        img_tensor = data['img'][0]  # DataContainer
        if debug:
            print(f"img_tensor type: {type(img_tensor)}")
        
        # 获取实际的张量数据
        img = img_tensor.data[0]  # 应该是包含图像数据的张量
        img_metas = data['img_metas'][0].data[0]  # 元信息
        
        if debug:
            print(f"img type: {type(img)}")
            if hasattr(img, 'shape'):
                print(f"img shape: {img.shape}")
            
            print(f"img_metas type: {type(img_metas)}")
            if isinstance(img_metas, list):
                print(f"img_metas length: {len(img_metas)}")
                print(f"First img_meta keys: {img_metas[0].keys() if isinstance(img_metas[0], dict) else 'not a dict'}")
            elif isinstance(img_metas, dict):
                print(f"img_metas keys: {img_metas.keys()}")
    
    except Exception as e:
        print(f"Error extracting data: {e}")
        return False
    
    # 获取图像归一化配置
    img_norm_cfg = None
    if isinstance(img_metas, list) and len(img_metas) > 0 and isinstance(img_metas[0], dict):
        img_norm_cfg = img_metas[0].get('img_norm_cfg')
    elif isinstance(img_metas, dict):
        img_norm_cfg = img_metas.get('img_norm_cfg')
    
    if img_norm_cfg and debug:
        print(f"img_norm_cfg: {img_norm_cfg}")
    
    # 获取预测结果
    if debug:
        print(f"Result keys: {result.keys()}")
    
    # 处理预测框
    pred_bboxes, pred_scores, pred_labels = None, None, None
    if 'pts_bbox' in result:
        if debug:
            print(f"pts_bbox keys: {result['pts_bbox'].keys()}")
        pred_bboxes = result['pts_bbox']['boxes_3d']
        pred_scores = result['pts_bbox']['scores_3d']
        pred_labels = result['pts_bbox']['labels_3d']
        
        # 根据得分阈值过滤
        mask = pred_scores > score_thr
        pred_bboxes = pred_bboxes[mask]
        pred_scores = pred_scores[mask]
        pred_labels = pred_labels[mask]
        
        print(f"\nNumber of objects detected: {len(pred_bboxes)}")
        if debug:
            print(f"Pred bboxes type: {type(pred_bboxes)}")
            if hasattr(pred_bboxes, 'tensor'):
                print(f"Pred bboxes tensor shape: {pred_bboxes.tensor.shape}")
                # 打印第一个检测框以进行调试
                if len(pred_bboxes) > 0:
                    print(f"First bbox: {pred_bboxes.tensor[0]}")
        print(f"Scores: {pred_scores.numpy()}")
        print(f"Labels: {pred_labels.numpy()}")
    else:
        print("No 'pts_bbox' in result")
    
    # 确定相机视图数量
    num_views = 1
    if isinstance(img, torch.Tensor):
        # 处理不同的张量形状
        if len(img.shape) == 5:  # [B, N, C, H, W]
            num_views = img.shape[1]
            if debug:
                print(f"Tensor shape is [B, N, C, H, W] with {num_views} views")
        elif len(img.shape) == 4:  # [N, C, H, W]
            num_views = img.shape[0]
            if debug:
                print(f"Tensor shape is [N, C, H, W] with {num_views} views")
        else:
            print(f"Unexpected img tensor shape: {img.shape}")
    else:
        print(f"img is not a tensor, it's {type(img)}")
        return False
    
    if debug:
        print(f"Number of camera views: {num_views}")
    
    # 清除所有子图，准备绘制新内容
    for ax in axes:
        ax.clear()
        ax.axis('off')
    
    # 定义类别颜色映射
    category_colors = [
        (0, 255, 0),    # 绿色
        (255, 0, 0),    # 蓝色 (BGR顺序)
        (0, 0, 255),    # 红色
        (255, 255, 0),  # 青色
        (0, 255, 255),  # 黄色
        (255, 0, 255),  # 品红色
        (125, 125, 0),  # 深蓝色
        (0, 125, 125),  # 暗黄色
        (125, 0, 125),  # 紫色
        (255, 125, 0),  # 淡蓝色
    ]
    
    # 处理每个相机视图
    for view in range(min(6, num_views)):
        try:
            # 统计投影状态
            stats = {
                'total': 0,
                'behind_camera': 0,
                'out_of_bounds': 0,
                'visible': 0,
                'failed': 0
            }
            
            # 根据图像形状获取相机图像
            if len(img.shape) == 5:  # [B, N, C, H, W]
                img_view = img[0, view].cpu()
            elif len(img.shape) == 4:  # [N, C, H, W]
                img_view = img[view].cpu()
            else:
                print(f"Skipping view {view} due to unexpected img shape")
                continue
            
            if debug:
                print(f"Camera {view} img_view shape: {img_view.shape}")
            
            # 反归一化图像
            img_np = denormalize_image(img_view, img_norm_cfg)
            
            if debug:
                print(f"Camera {view} img_np shape after processing: {img_np.shape}")
                if img_norm_cfg:
                    print(f"to_rgb flag: {img_norm_cfg.get('to_rgb', True)}")
                # 检查图像是否为连续数组
                print(f"Image is contiguous: {img_np.flags['C_CONTIGUOUS']}")
            
            # 获取该视图的投影矩阵
            lidar2img = get_projection_matrix(img_metas, view)
            
            if debug:
                if lidar2img is not None:
                    print(f"lidar2img for view {view} found: {type(lidar2img)}")
                    if hasattr(lidar2img, 'shape'):
                        print(f"lidar2img shape after conversion: {lidar2img.shape}")
                else:
                    print(f"lidar2img for view {view} not found")
            
            # 如果有预测结果和投影矩阵，添加可视化
            visible_boxes = 0
            if pred_bboxes is not None and lidar2img is not None and len(pred_bboxes) > 0:
                try:
                    # 绘制检测框和标签
                    centers_3d = pred_bboxes.gravity_center.numpy()
                    stats['total'] = len(centers_3d)
                    
                    for i, center in enumerate(centers_3d):
                        try:
                            # 获取目标类别对应的颜色
                            cat_id = int(pred_labels[i].item())
                            color = category_colors[cat_id % len(category_colors)]
                            
                            # 投影3D中心点到图像
                            center_4d = np.append(center, 1)
                            
                            # 处理不同的lidar2img格式
                            if isinstance(lidar2img, np.ndarray) and lidar2img.ndim == 2 and lidar2img.shape[0] == 4:
                                # 标准4x4投影矩阵
                                center_2d = np.dot(lidar2img, center_4d)
                            elif isinstance(lidar2img, np.ndarray) and lidar2img.ndim == 3:
                                # 多个投影矩阵(使用第一个)
                                center_2d = np.dot(lidar2img[0], center_4d)
                            else:
                                if debug:
                                    print(f"Unexpected lidar2img format for detection {i}")
                                continue
                            
                            # 如果在相机后方则跳过
                            if center_2d[2] <= 0:
                                if debug:
                                    print(f"Center {i} is behind camera (z={center_2d[2]})")
                                stats['behind_camera'] += 1
                                continue
                            
                            # 齐次除法
                            center_2d_x = center_2d[0] / center_2d[2]
                            center_2d_y = center_2d[1] / center_2d[2]
                            
                            # 检查点是否在图像范围内
                            img_h, img_w = img_np.shape[:2]
                            if 0 <= center_2d_x < img_w and 0 <= center_2d_y < img_h:
                                visible_boxes += 1
                                stats['visible'] += 1
                                
                                try:
                                    # 绘制中心点圆圈
                                    cv2.circle(img_np, 
                                              (int(center_2d_x), int(center_2d_y)), 
                                              radius=10, 
                                              color=color, 
                                              thickness=-1)
                                    
                                    # 添加标签文本（带黑色边框）
                                    label_str = f"{cat_id}: {pred_scores[i].item():.2f}"
                                    cv2.putText(img_np, 
                                               label_str, 
                                               (int(center_2d_x)+10, int(center_2d_y)-10),
                                               cv2.FONT_HERSHEY_SIMPLEX, 
                                               1.0, 
                                               (0, 0, 0), 
                                               3)  # 黑色边框
                                    cv2.putText(img_np, 
                                               label_str, 
                                               (int(center_2d_x)+10, int(center_2d_y)-10),
                                               cv2.FONT_HERSHEY_SIMPLEX, 
                                               1.0, 
                                               color, 
                                               2)  # 彩色文本
                                except Exception as e:
                                    if debug:
                                        print(f"Error drawing text/circle for detection {i}: {e}")
                                    stats['failed'] += 1
                                
                                # 尝试绘制3D边界框
                                try:
                                    corners_3d = pred_bboxes.corners[i].numpy()
                                    
                                    # 投影所有角点到图像空间
                                    corners_2d_list = []
                                    for corner_idx in range(8):
                                        corner = corners_3d[corner_idx]
                                        corner_4d = np.append(corner, 1)
                                        
                                        # 处理不同的lidar2img格式
                                        if isinstance(lidar2img, np.ndarray) and lidar2img.ndim == 2 and lidar2img.shape[0] == 4:
                                            corner_2d = np.dot(lidar2img, corner_4d)
                                        elif isinstance(lidar2img, np.ndarray) and lidar2img.ndim == 3:
                                            corner_2d = np.dot(lidar2img[0], corner_4d)
                                        else:
                                            if debug and corner_idx == 0:
                                                print(f"Unexpected lidar2img format for box {i}")
                                            break
                                        
                                        # 如果角点在相机后方则跳过
                                        if corner_2d[2] <= 0:
                                            if debug and corner_idx == 0:
                                                print(f"Corner {corner_idx} of box {i} is behind camera")
                                            break
                                        
                                        # 齐次除法
                                        corner_2d = corner_2d[:2] / corner_2d[2]
                                        corners_2d_list.append(corner_2d)
                                    
                                    # 如果有所有8个角点，绘制边界框
                                    if len(corners_2d_list) == 8:
                                        try:
                                            draw_3d_box(img_np, np.array(corners_2d_list), color=color, thickness=2)
                                        except Exception as e:
                                            if debug:
                                                print(f"Error drawing 3D box for detection {i}: {e}")
                                            stats['failed'] += 1
                                    else:
                                        if debug:
                                            print(f"Could not project all corners of box {i} to 2D (got {len(corners_2d_list)} corners)")
                                except Exception as e:
                                    if debug:
                                        print(f"Error processing corners for detection {i}: {e}")
                                    stats['failed'] += 1
                            else:
                                if debug:
                                    print(f"Center {i} out of bounds: ({center_2d_x}, {center_2d_y}) not in ({img_w}, {img_h})")
                                stats['out_of_bounds'] += 1
                                
                                # 绘制中心点圆圈 (即使在视野外，也在图像边缘标记)
                                edge_x = min(max(0, int(center_2d_x)), img_w-1)
                                edge_y = min(max(0, int(center_2d_y)), img_h-1)
                                
                                try:
                                    cv2.circle(img_np, 
                                             (edge_x, edge_y), 
                                             radius=5, 
                                             color=(0, 0, 255),  # 红色表示视野外的目标
                                             thickness=-1)
                                except Exception as e:
                                    if debug:
                                        print(f"Error drawing out-of-bounds marker for detection {i}: {e}")
                                    stats['failed'] += 1
                        
                        except Exception as e:
                            if debug:
                                print(f"Error drawing detection {i}: {e}")
                            stats['failed'] += 1
                    
                    # 在相机画面上添加可见物体计数和统计信息
                    try:
                        stats_text = f"Objects: {visible_boxes}/{stats['total']}"
                        cv2.putText(img_np, 
                                  stats_text, 
                                  (20, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  1.0, 
                                  (0, 0, 0), 
                                  3)
                        cv2.putText(img_np, 
                                  stats_text, 
                                  (20, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  1.0, 
                                  (255, 255, 255), 
                                  2)
                        
                        # 添加详细统计
                        detail_stats = f"Behind:{stats['behind_camera']} OutOfBounds:{stats['out_of_bounds']} Visible:{stats['visible']}"
                        cv2.putText(img_np, 
                                  detail_stats, 
                                  (20, 70),
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, 
                                  (0, 0, 0), 
                                  3)
                        cv2.putText(img_np, 
                                  detail_stats, 
                                  (20, 70),
                                  cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, 
                                  (255, 255, 255), 
                                  1)
                    except Exception as e:
                        print(f"Error drawing stats text for view {view}: {e}")
                    
                except Exception as e:
                    print(f"Error processing detections for view {view}: {e}")
            elif debug:
                if pred_bboxes is None:
                    print(f"No predictions for view {view}")
                elif lidar2img is None:
                    print(f"No projection matrix for view {view}")
                elif len(pred_bboxes) == 0:
                    print(f"No detections passed the score threshold for view {view}")
            
            # 更新子图
            axes[view].imshow(img_np)
            axes[view].set_title(f'Camera {view} ({visible_boxes} obj)')
            axes[view].axis('off')
        except Exception as e:
            print(f"Error processing camera view {view}: {e}")
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    
    # 保存可视化结果
    if out_dir:
        # 获取帧ID用于文件命名
        frame_id = get_frame_id(img_metas)
        
        # 使用帧ID或批次索引
        if frame_id is None:
            out_file = osp.join(out_dir, f'frame_{batch_idx:06d}.png')
        else:
            if isinstance(frame_id, (int, float)):
                out_file = osp.join(out_dir, f'frame_{int(frame_id):06d}.png')
            else:
                # 对于字符串ID
                out_file = osp.join(out_dir, f'frame_{frame_id}.png')
        
        plt.savefig(out_file)
        print(f"Visualization saved to {out_file}")
    
    # 等待用户输入
    if show:
        key = input("\nPress Enter for next frame, 'q' to quit: ")
        if key.lower() == 'q':
            return True  # 发出停止可视化的信号
    
    return False  # 继续可视化

def get_frame_id(img_metas):
    """
    从元数据中提取帧ID
    
    Args:
        img_metas: 图像元数据
        
    Returns:
        int | str | None: 帧ID
    """
    frame_id = None
    
    # 尝试从元数据获取帧ID
    if isinstance(img_metas, list) and len(img_metas) > 0:
        if isinstance(img_metas[0], dict):
            if 'frame_idx' in img_metas[0]:
                frame_id = img_metas[0]['frame_idx']
            elif 'sample_idx' in img_metas[0]:
                frame_id = img_metas[0]['sample_idx']
    elif isinstance(img_metas, dict):
        if 'frame_idx' in img_metas:
            frame_id = img_metas['frame_idx']
        elif 'sample_idx' in img_metas:
            frame_id = img_metas['sample_idx']
    
    return frame_id

def main():
    """脚本主函数"""
    args = parse_args()
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    # 构建模型
    model = build_model(cfg.model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    # 设置数据加载器配置
    cfg.data.test.test_mode = True
    cfg.data.test.samples_per_gpu = 1
    cfg.data.test.workers_per_gpu = 0
    
    # 构建数据集
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False
    )
    
    # 设置模型为评估模式
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    
    # 创建输出目录
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    # 创建图形和轴（以便重用）
    plt.ion()
    fig = plt.figure(figsize=(24, 16))
    fig.canvas.manager.set_window_title('BEVFormer Multi-View Visualization')
    
    # 创建子图
    axes = []
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        ax.set_title(f'Camera {i}')
        ax.axis('off')
        axes.append(ax)
    
    # 处理每个批次
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        # 调用自定义可视化函数
        should_stop = visualize_multi_modality_result(
            data, result[0], args.out_dir, fig, axes, args.score_thr, args.debug, i
        )
        if should_stop:
            break
    
    # 结束时关闭图形
    plt.close(fig)

if __name__ == '__main__':
    main() 