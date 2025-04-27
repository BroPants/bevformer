import os
import sys
import glob
sys.path.append('.')  # 添加当前目录到Python路径

from nuscenes.nuscenes import NuScenes
from tools.analysis_tools.visual import render_sample_data, BoxVisibility, get_sample_data
import matplotlib.pyplot as plt
import time
import shutil
import cv2
import numpy as np

try:
    # 创建临时保存目录和最终目录
    temp_dir = 'visualization_results'
    camera_views_dir = 'camera_views'
    bird_eye_views_dir = 'bird_eye_views'
    
    # 清理并创建临时目录
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # 初始化NuScenes数据集
    nusc = NuScenes(
        version='v1.0-mini',
        dataroot='data/nuscenes',
        verbose=True
    )
    
    # 所有相机名称
    CAMERA_SENSORS = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_FRONT_LEFT'
    ]
    
    # 获取数据集中的样本总数
    total_samples = len(nusc.sample)
    print(f"Processing first {total_samples} samples")
    
    # 记录成功和失败的样本
    successful_samples = 0
    failed_samples = []
    
    # 遍历样本并可视化
    for i, sample in enumerate(nusc.sample[:total_samples]):
        sample_token = sample['token']
        print(f"Processing sample {i+1}/{total_samples}, token: {sample_token}")
        
        try:
            # 创建一个大图来存放所有相机视图
            plt.figure(figsize=(24, 16))
            
            # 遍历所有相机
            for idx, camera in enumerate(CAMERA_SENSORS):
                # 获取相机数据
                sample_data = nusc.get('sample_data', sample['data'][camera])
                if sample_data is None:
                    print(f"Warning: No {camera} data for sample {i+1}")
                    continue
                    
                # 读取相机图像
                img_path = os.path.join(nusc.dataroot, sample_data['filename'])
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read image from {img_path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 在大图中添加这个相机视图
                plt.subplot(2, 3, idx + 1)
                plt.title(camera.replace('CAM_', ''), fontsize=12)
                plt.imshow(img)
                plt.axis('off')
            
            # 调整子图之间的间距
            plt.tight_layout()
            
            # 保存相机视图组合图到临时目录
            camera_view_path = os.path.join(temp_dir, f'sample_{i+1:04d}_cams.png')
            plt.savefig(camera_view_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            # 渲染样本数据 - 俯视图
            plt.figure(figsize=(16, 9))
            bird_eye_view_path = os.path.join(temp_dir, f'sample_{i+1:04d}.png')
            
            render_sample_data(
                sample_token,
                with_anns=True,
                box_vis_level=BoxVisibility.ANY,
                axes_limit=40,
                out_path=bird_eye_view_path,
                nusc=nusc,
                verbose=False
            )
            plt.close()
            
            # 检查文件是否成功创建
            if os.path.exists(camera_view_path) and os.path.exists(bird_eye_view_path):
                successful_samples += 1
                print(f"✓ Successfully saved visualization for sample {i+1}")
            else:
                failed_samples.append(i+1)
                print(f"✗ Failed to save visualization for sample {i+1}")
                
        except Exception as e:
            failed_samples.append(i+1)
            print(f"✗ Error processing sample {i+1}: {str(e)}")
            plt.close('all')  # 确保所有图形都被关闭
        
        # 每处理5个样本打印一次进度
        if (i+1) % 5 == 0 or (i+1) == total_samples:
            print(f"Progress: {i+1}/{total_samples} samples processed")
            print(f"Successful: {successful_samples}, Failed: {len(failed_samples)}")
    
    print("\nRendering completed. Now organizing files...")
    
    # 创建最终目录
    os.makedirs(camera_views_dir, exist_ok=True)
    os.makedirs(bird_eye_views_dir, exist_ok=True)
    
    # 移动文件到对应目录
    for filename in os.listdir(temp_dir):
        src_path = os.path.join(temp_dir, filename)
        if filename.endswith('_cams.png'):
            dst_path = os.path.join(camera_views_dir, filename)
        else:
            dst_path = os.path.join(bird_eye_views_dir, filename)
        shutil.move(src_path, dst_path)
    
    # 删除临时目录
    shutil.rmtree(temp_dir)
    
    # 打印最终结果
    print("\n" + "="*50)
    print(f"Visualization completed. Results saved to:")
    print(f"1. {camera_views_dir}/ - Camera views")
    print(f"2. {bird_eye_views_dir}/ - Bird's eye views")
    print(f"Successfully processed: {successful_samples}/{total_samples} samples")
    
    if failed_samples:
        print(f"Failed samples: {failed_samples}")
        print("You may want to retry processing these samples individually.")
    else:
        print("All samples were successfully processed!")
    
    # 创建一个简单的HTML文件来查看结果
    html_path = 'visualization_results.html'
    with open(html_path, 'w') as f:
        f.write('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>NuScenes Mini Dataset Visualization</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333; }
                .container { display: flex; flex-wrap: wrap; }
                .sample { margin: 10px; padding: 10px; border: 1px solid #ddd; width: 100%; }
                .sample img { max-width: 100%; height: auto; }
                .sample h3 { margin-top: 0; }
            </style>
        </head>
        <body>
            <h1>NuScenes Mini Dataset Visualization</h1>
            <p>Total samples: ''' + str(total_samples) + '''</p>
            <p>Successfully processed: ''' + str(successful_samples) + '''</p>
            <div class="container">
        ''')
        
        # 添加所有样本的可视化结果
        for i in range(1, total_samples + 1):
            if i not in failed_samples:
                f.write(f'''
                <div class="sample">
                    <h3>Sample {i:04d}</h3>
                    <h4>Camera Views</h4>
                    <img src="{camera_views_dir}/sample_{i:04d}_cams.png" alt="Sample {i} camera views">
                    <h4>Bird's Eye View</h4>
                    <img src="{bird_eye_views_dir}/sample_{i:04d}.png" alt="Sample {i} bird's eye view">
                </div>
                ''')
        
        f.write('''
            </div>
        </body>
        </html>
        ''')
    
    print(f"Created HTML viewer at {html_path}")

except Exception as e:
    print(f"Error occurred: {str(e)}")
    raise 