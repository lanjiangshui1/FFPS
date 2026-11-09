_base_ = [
    './_base_/custom_nus-3d.py',
    './_base_/default_runtime.py'
]

sync_bn = True
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
camera_used = ['left']
# load_from = '/mnt/hdddata2/lhj/ckpts/iou47.15_miou19.03.pth'

# 20 classes with unlabeled
class_names = [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
    'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
    'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
    'pole', 'traffic-sign',
]
num_class = len(class_names)

point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
occ_size = [256, 256, 32]
# downsample ratio in [x, y, z] when generating 3D volumes in LSS
lss_downsample = [2, 2, 2]

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
voxel_size = [voxel_x, voxel_y, voxel_z]

data_config = {
    'input_size': (384, 1280),
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x ],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y ],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z ],
    'dbound': [2.0, 58.0, 0.5],
}

target_frames = [-3, -6, -9, -12]
numC_Trans = 192
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    type='FFPS',
    n_classes=num_class,
    img_backbone=dict(
       type='ResNet',
       depth=50,
       num_stages=4,
       out_indices=(0,1,2),
       frozen_stages=-1,
       norm_cfg=dict(type='BN', requires_grad=False),
       norm_eval=True,
       style='pytorch',
       with_cp=True,
       pretrained=dict(img='ckpts/resnet50-19c8e357.pth')),
    img_neck=dict(
        type='MSDeformAttnPixelDecoder',
        in_channels=[256, 512, 1024],
        aggregate_target=2,
        out_channels=[192, 192, 192]),
    img_view_transformer=dict(
        type='OAProjectionFlow',
        mlvl_nums=3,
        grid_config=grid_config,
        data_config=data_config,
        numC=numC_Trans,
        cam_channels=33),
    img_bev_encoder_backbone=None,
    img_bev_encoder_neck=None,
    flow_encoder_backbone=dict(
        type='CustomResNet3D',
        depth=18,
        n_input_channels=384,
        block_inplanes=[64, 128, 256, 512],
        out_indices=(0, 1, 2, 3),
        norm_cfg=norm_cfg,
    ),
    flow_predictor=dict(
        type='Predictor',
        n_input_channels=[64, 128, 256, 512],
        in_timesteps=2,
        out_timesteps=1,
        norm_cfg=norm_cfg,
    ),
    flow_encoder_neck=dict(
        type='FPN3D',
        in_channels=[64, 128, 256, 512],
        out_channels=192,
        norm_cfg=norm_cfg,
    ),
    flow_head=dict(
        type='FlowHead',
        num_level=1,
        in_channels=[192],
        out_channel=3,
        soft_weights=True,
        norm_cfg=norm_cfg,
    ),
    loss_weight_cfg=dict(
        loss_voxel_ce_weight=1.0,
        loss_voxel_sem_scal_weight=1.0,
        loss_voxel_geo_scal_weight=1.0,
        loss_voxel_scan_weight=0.5,
    ),
    aux_loss_weight=0.4,
    loss_align_weight=0.05,
)

dataset_type = 'CustomSSCBenchLssDataset'
data_root = 'data/SSCBench-KITTI360'
ann_file = 'data/SSCBench-KITTI360/preprocess'
depth_path = 'data/SSCBench-KITTI360/stereodepth_benchv2'


bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    flip_dz_ratio=0.5,)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_SemanticKitti', is_train=True,
            data_config=data_config, img_norm_cfg=img_norm_cfg,
            stereo_path=depth_path, dataset='bench'),
    dict(type='LoadSemKittiAnnotation', bda_aug_conf=bda_aug_conf, 
            is_train=True, point_cloud_range=point_cloud_range),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ'], 
            meta_keys=['pc_range', 'occ_size']),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_SemanticKitti', is_train=False, 
         data_config=data_config, img_norm_cfg=img_norm_cfg,
         stereo_path=depth_path, dataset='bench'),
    dict(type='LoadSemKittiAnnotation', bda_aug_conf=bda_aug_conf,
            is_train=False, point_cloud_range=point_cloud_range),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=False), 
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ'], 
            meta_keys=['pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img']),
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

test_config=dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=ann_file,
    pipeline=test_pipeline,
    classes=class_names,
    modality=input_modality,
    split='test',
    camera_used=camera_used,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    target_frames = target_frames,
)

val_config=dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=ann_file,
    pipeline=test_pipeline,
    classes=class_names,
    modality=input_modality,
    split='val',
    test_mode=False,
    camera_used=camera_used,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    target_frames = target_frames,
)


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        split='train',
        camera_used=camera_used,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        target_frames = target_frames,
    ),
    val=val_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

# for most of these optimizer settings, we follow Mask2Former
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.01,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
        },
        norm_decay_mult=0.0))

optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
# optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    step=[20, 25],
)

# checkpoint_config = dict(max_keep_ckpts=1, interval=1)
checkpoint_config = dict(interval=1)
runner = dict(type='EpochBasedRunner', max_epochs=30)

evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='semkitti_SSC_mIoU',
    rule='greater',
)

