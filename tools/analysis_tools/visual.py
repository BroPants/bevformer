# Based on https://github.com/nutonomy/nuscenes-devkit
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import mmcv
from nuscenes.nuscenes import NuScenes
from PIL import Image
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from typing import Tuple, List, Iterable
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.render import visualize_sample




cams = ['CAM_FRONT',
 'CAM_FRONT_RIGHT',
 'CAM_BACK_RIGHT',
 'CAM_BACK',
 'CAM_BACK_LEFT',
 'CAM_FRONT_LEFT']

import numpy as np
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from PIL import Image
from matplotlib import rcParams


def render_annotation(
        anntoken: str,
        margin: float = 10,
        view: np.ndarray = np.eye(4),
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        out_path: str = 'render.png',
        extra_info: bool = False) -> None:
    """
    Render selected annotation.
    :param anntoken: Sample_annotation token.
    :param margin: How many meters in each direction to include in LIDAR view.
    :param view: LIDAR view point.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param out_path: Optional path to save the rendered figure to disk.
    :param extra_info: Whether to render extra information below camera view.
    """
    ann_record = nusc.get('sample_annotation', anntoken)
    sample_record = nusc.get('sample', ann_record['sample_token'])
    assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'

    # Figure out which camera the object is fully visible in (this may return nothing).
    boxes, cam = [], []
    cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
    all_bboxes = []
    select_cams = []
    for cam in cams:
        _, boxes, _ = nusc.get_sample_data(sample_record['data'][cam], box_vis_level=box_vis_level,
                                           selected_anntokens=[anntoken])
        if len(boxes) > 0:
            all_bboxes.append(boxes)
            select_cams.append(cam)
            # We found an image that matches. Let's abort.
    # assert len(boxes) > 0, 'Error: Could not find image where annotation is visible. ' \
    #                      'Try using e.g. BoxVisibility.ANY.'
    # assert len(boxes) < 2, 'Error: Found multiple annotations. Something is wrong!'

    num_cam = len(all_bboxes)

    fig, axes = plt.subplots(1, num_cam + 1, figsize=(18, 9))
    select_cams = [sample_record['data'][cam] for cam in select_cams]
    print('bbox in cams:', select_cams)
    # Plot LIDAR view.
    lidar = sample_record['data']['LIDAR_TOP']
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(lidar, selected_anntokens=[anntoken])
    LidarPointCloud.from_file(data_path).render_height(axes[0], view=view)
    for box in boxes:
        c = np.array(get_color(box.name, nusc)) / 255.0
        box.render(axes[0], view=view, colors=(c, c, c))
        corners = view_points(boxes[0].corners(), view, False)[:2, :]
        axes[0].set_xlim([np.min(corners[0, :]) - margin, np.max(corners[0, :]) + margin])
        axes[0].set_ylim([np.min(corners[1, :]) - margin, np.max(corners[1, :]) + margin])
        axes[0].axis('off')
        axes[0].set_aspect('equal')

    # Plot CAMERA view.
    for i in range(1, num_cam + 1):
        cam = select_cams[i - 1]
        data_path, boxes, camera_intrinsic = nusc.get_sample_data(cam, selected_anntokens=[anntoken])
        im = Image.open(data_path)
        axes[i].imshow(im)
        axes[i].set_title(nusc.get('sample_data', cam)['channel'])
        axes[i].axis('off')
        axes[i].set_aspect('equal')
        for box in boxes:
            c = np.array(get_color(box.name, nusc)) / 255.0
            box.render(axes[i], view=camera_intrinsic, normalize=True, colors=(c, c, c))

        # Print extra information about the annotation below the camera view.
        axes[i].set_xlim(0, im.size[0])
        axes[i].set_ylim(im.size[1], 0)

    if extra_info:
        rcParams['font.family'] = 'monospace'

        w, l, h = ann_record['size']
        category = ann_record['category_name']
        lidar_points = ann_record['num_lidar_pts']
        radar_points = ann_record['num_radar_pts']

        sample_data_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        dist = np.linalg.norm(np.array(pose_record['translation']) - np.array(ann_record['translation']))

        information = ' \n'.join(['category: {}'.format(category),
                                  '',
                                  '# lidar points: {0:>4}'.format(lidar_points),
                                  '# radar points: {0:>4}'.format(radar_points),
                                  '',
                                  'distance: {:>7.3f}m'.format(dist),
                                  '',
                                  'width:  {:>7.3f}m'.format(w),
                                  'length: {:>7.3f}m'.format(l),
                                  'height: {:>7.3f}m'.format(h)])

        plt.annotate(information, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')

    if out_path is not None:
        plt.savefig(out_path)



def get_sample_data(
        sample_data_token: str,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        selected_anntokens=None,
        use_flat_vehicle_coordinates: bool = False,
        nusc=None):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :param nusc: NuScenes object.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """
    if nusc is None:
        raise ValueError("nusc parameter is required")

    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue

        box_list.append(box)

    return data_path, box_list, cam_intrinsic



def get_predicted_data(sample_data_token: str,
                       box_vis_level: BoxVisibility = BoxVisibility.ANY,
                       selected_anntokens=None,
                       use_flat_vehicle_coordinates: bool = False,
                       pred_anns=None
                       ):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    # if selected_anntokens is not None:
    #    boxes = list(map(nusc.get_box, selected_anntokens))
    # else:
    #    boxes = nusc.get_boxes(sample_data_token)
    boxes = pred_anns
    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue
        box_list.append(box)

    return data_path, box_list, cam_intrinsic




def lidiar_render(sample_token, data, out_path=None, nusc=None):
    bbox_gt_list = []
    bbox_pred_list = []
    
    # Get ground truth boxes
    sample = nusc.get('sample', sample_token)
    anns = sample['anns']
    for ann in anns:
        ann_record = nusc.get('sample_annotation', ann)
        bbox_gt = nusc.get_box(ann_record['token'])
        bbox_gt_list.append(bbox_gt)
    
    # Get predicted boxes if available
    if data is not None and 'results' in data:
        bbox_anns = data['results'][sample_token]
        for box in bbox_anns:
            bbox_pred = DetectionBox(
                sample_token=box['sample_token'],
                translation=box['translation'],
                size=box['size'],
                rotation=box['rotation'],
                velocity=box['velocity'],
                detection_name=box['detection_name'],
                detection_score=box['detection_score'],
                attribute_name=box['attribute_name']
            )
            bbox_pred_list.append(bbox_pred)
    
    # Visualize
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    
    # Get and render LIDAR data
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_path = nusc.get_sample_data_path(lidar_token)
    lidar_pc = LidarPointCloud.from_file(lidar_path)
    points = lidar_pc.points[:3, :]  # Get x, y, z coordinates
    
    # Plot points with a colormap based on height (z-coordinate)
    scatter = ax.scatter(points[0, :], points[1, :], c=points[2, :],
                        cmap='viridis', s=0.2, alpha=0.5)
    plt.colorbar(scatter, label='Height')
    
    # Show ground truth boxes
    for box in bbox_gt_list:
        c = np.array(get_color(box.name, nusc)) / 255.0
        box.render(ax, view=np.eye(4), colors=(c, c, c))
    
    # Show predicted boxes
    for box in bbox_pred_list:
        c = np.array(get_color(box.detection_name, nusc)) / 255.0
        box.render(ax, view=np.eye(4), colors=(c, c, c))
    
    # Set axes limits and title
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_title('Bird\'s Eye View')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.grid(True)
    
    # Save or show
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def get_color(category_name: str, nusc=None) -> List[int]:
    """
    Get color for a category.
    :param category_name: Category name.
    :param nusc: NuScenes object (not used).
    :return: List of RGB values.
    """
    # Default colors for all categories
    colors = {
        'car': [0, 255, 0],  # Green
        'truck': [0, 0, 255],  # Blue
        'bus': [255, 0, 0],  # Red
        'trailer': [255, 128, 0],  # Orange
        'construction_vehicle': [128, 0, 255],  # Purple
        'pedestrian': [255, 255, 0],  # Yellow
        'motorcycle': [0, 255, 255],  # Cyan
        'bicycle': [255, 0, 255],  # Magenta
        'traffic_cone': [128, 128, 128],  # Gray
        'barrier': [64, 64, 64],  # Dark Gray
        # Add more categories as needed
        'human.pedestrian.adult': [255, 255, 0],  # Yellow
        'human.pedestrian.child': [255, 255, 128],  # Light Yellow
        'human.pedestrian.construction_worker': [255, 255, 64],  # Dark Yellow
        'human.pedestrian.police_officer': [255, 255, 192],  # Very Light Yellow
        'vehicle.car': [0, 255, 0],  # Green
        'vehicle.truck': [0, 0, 255],  # Blue
        'vehicle.bus.bendy': [255, 0, 0],  # Red
        'vehicle.bus.rigid': [255, 64, 64],  # Light Red
        'vehicle.motorcycle': [0, 255, 255],  # Cyan
        'vehicle.bicycle': [255, 0, 255],  # Magenta
        'vehicle.trailer': [255, 128, 0],  # Orange
        'movable_object.barrier': [64, 64, 64],  # Dark Gray
        'movable_object.trafficcone': [128, 128, 128],  # Gray
    }
    
    # Try to match the exact category name first
    if category_name in colors:
        return colors[category_name]
    
    # Try to match by lowercase comparison
    category_lower = category_name.lower()
    for key, value in colors.items():
        if category_lower == key.lower():
            return value
    
    # Return black if no match found
    return [0, 0, 0]


def render_sample_data(
        sample_toekn: str,
        with_anns: bool = True,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        axes_limit: float = 40,
        ax=None,
        nsweeps: int = 1,
        out_path: str = None,
        underlay_map: bool = True,
        use_flat_vehicle_coordinates: bool = True,
        show_lidarseg: bool = False,
        show_lidarseg_legend: bool = False,
        filter_lidarseg_labels=None,
        lidarseg_preds_bin_path: str = None,
        verbose: bool = True,
        show_panoptic: bool = False,
        pred_data=None,
        nusc=None,
      ) -> None:
    """
    Render sample data.
    :param sample_toekn: Sample token.
    :param with_anns: Whether to draw box annotations.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param axes_limit: Axes limit for the visualization.
    :param ax: Axes to draw on.
    :param nsweeps: Number of sweeps to render.
    :param out_path: Optional path to save the rendered figure to disk.
    :param underlay_map: When set to true, LIDAR data is plotted onto the map.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :param show_lidarseg: Whether to show lidar segmentation.
    :param show_lidarseg_legend: Whether to show lidar segmentation legend.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes.
    :param lidarseg_preds_bin_path: Path to the lidar segmentation predictions binary file.
    :param verbose: Whether to print information about the sample.
    :param show_panoptic: Whether to show panoptic segmentation.
    :param pred_data: Predicted data.
    :param nusc: NuScenes object.
    """
    if nusc is None:
        raise ValueError("nusc parameter is required")
        
    # Call the lidar render function
    lidiar_render(sample_toekn, pred_data, out_path=out_path, nusc=nusc)
    
    # Get sample data
    sample = nusc.get('sample', sample_toekn)
    
    # Get camera data
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    ax = ax.flatten()
    
    for i, cam in enumerate(cams):
        data_path, boxes, camera_intrinsic = get_sample_data(sample['data'][cam], box_vis_level=box_vis_level, nusc=nusc)
        im = Image.open(data_path)
        ax[i].imshow(im)
        ax[i].set_title(cam)
        ax[i].axis('off')
        
        # Draw 3D bounding boxes
        if with_anns:
            for box in boxes:
                c = np.array(get_color(box.name, nusc)) / 255.0
                box.render(ax[i], view=camera_intrinsic, normalize=True, colors=(c, c, c))
    
    # Save or show
    if out_path:
        plt.savefig(out_path.replace('.png', '_cams.png'))
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)
    # render_annotation('7603b030b42a4b1caa8c443ccc1a7d52')
    bevformer_results = mmcv.load('test/bevformer_base/Thu_Jun__9_16_22_37_2022/pts_bbox/results_nusc.json')
    sample_token_list = list(bevformer_results['results'].keys())
    for id in range(0, 10):
        render_sample_data(sample_token_list[id], pred_data=bevformer_results, out_path=sample_token_list[id])
