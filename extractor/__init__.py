# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

# Adapted by Olivier Jansen

import os
import platform

# Hacky-tacky way to use EGL only on the headless VM (which runs Linux) but not on my MacBook (Darwin)
if platform.system() == 'Linux':
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import time
import torch
import pickle
import shutil
import colorsys
import argparse
import glob
import numpy as np
from multi_person_tracker import MPT
from checkpoints_loader import CheckpointsLoader
from torch.utils.data import DataLoader
import logging
import subprocess

from .lib.models.vibe import PoseGenerator
from .lib.utils.renderer import Renderer
from .lib.dataset.inference import Inference
from .lib.data_utils.kp_utils import convert_kps
from .lib.utils.pose_tracker import run_posetracker

from .lib.utils.demo_utils import (
    smplify_runner,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    images_to_video
)

MIN_NUM_FRAMES = 25
BBOX_SCALE = 1.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SingleVideoExtractor():
    results = {}

    def __init__(self, video_file, pretrained_spin, tracking_method='yolo'):
        # Path to video file
        self.video_file = video_file
        if not os.path.isabs(self.video_file):
            self.video_file = os.path.join(os.getcwd(), self.video_file)

        # Tracking method (can be YOLOv3, Openpose or MaskRCNN)
        self.tracking_method = tracking_method
        self.pretrained_spin = pretrained_spin

        # Name extracted from video filename
        # TODO: maybe replace by GUID?
        self.name = os.path.basename(self.video_file).replace('.mp4', '').replace('.', '_')

        # Create a folder in /tmp for frame images
        self.image_folder = os.path.join('/tmp', self.name)
        os.makedirs(self.image_folder, exist_ok=True)


    def run(self, output_folder, render=False):
        logging.info(f'Running extraction for video \'{self.name}\'')

        # Split video into images
        self._video_to_images()
        # Find person bounding boxes or 2D joints to make crops from
        self._find_tracklets()
        # Run VIBE on these tracklets
        self._run_vibe()
        # Save results to the output folder
        self._save(output_folder)
        # Render the results (videos with bboxes and 3D poses) to output folder
        if render:
            self._render(output_folder)
        # Remove temporary data
        self._clean()

    def _video_to_images(self):
        """Split the video into frame images using ffmpeg"""

        # Use ffmpeg to split video into frames
        command = ['ffmpeg',
                '-i', self.video_file,
                '-f', 'image2',
                '-v', 'error',
                f'{self.image_folder}/%06d.png']

        subprocess.call(command, stdout=subprocess.DEVNULL)

        image_shape = cv2.imread(os.path.join(self.image_folder, '000001.png')).shape
        self.orig_height, self.orig_width = image_shape[:2]


    def _save(self, output_folder, rendered_tracklets=False, rendered_motion=False):
        # Create a folder for the output pickles
        self.output_path = os.path.join(self.output_folder, name)
        os.makedirs(self.output_path, exist_ok=True)

        for person in self.results.keys():
            dump_path = os.path.join(output_path, "%s.pkl" % person)
            pickle.dump(self.results[person], open(dump_path, 'wb'))

    def _render_bbox(self, output_folder):
        pass

    def _render_vibe(self, output_folder):
        renderer = Renderer(resolution=(self.orig_width, self.orig_height), orig_img=True)

        output_img_folder = f'{self.image_folder}_output'
        os.makedirs(output_img_folder, exist_ok=True)

        # prepare results for rendering
        num_frames = len(os.listdir(self.image_folder))
        frame_results = prepare_rendering_results(self.results, num_frames)
        mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in self.results.keys()}

        image_file_names = sorted([
            os.path.join(self.image_folder, x)
            for x in os.listdir(self.image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])

        for frame_idx in range(len(image_file_names)):
            img_fname = image_file_names[frame_idx]
            img = cv2.imread(img_fname)

            for person_id, person_data in frame_results[frame_idx].items():
                frame_verts = person_data['verts']
                frame_cam = person_data['cam']

                mc = mesh_color[person_id]

                mesh_filename = None

                img = renderer.render(
                    img,
                    frame_verts,
                    cam=frame_cam,
                    color=mc,
                    mesh_filename=mesh_filename,
                )

            cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img)

        # ========= Save rendered video ========= #
        vid_name = os.path.basename(self.video_file)
        save_name = 'vibe.mp4'
        save_name = os.path.join(output_path, save_name)
        # print(f'Saving result video to {save_name}')
        images_to_video(img_folder=output_img_folder, output_vid_file=save_name)
        shutil.rmtree(output_img_folder)

    def _find_tracklets(self):
        if self.tracking_method == 'openpose':
            # Use OpenPose Spatio-Temporal Affinity Fields to find 2D poses in video
            tracking_results = run_posetracker(self.video_file, staf_folder=self.staf_dir)

        elif self.tracking_method == 'yolo' or self.tracking_method == 'maskrcnn':
            # Use Multi-Person Tracker with YOLOv3 and SORT to find bounding boxes in video
            mpt = MPT(
                device=device,
                detector_type=self.tracking_method,
                output_format='dict',
                yolo_img_size=416,
            )
            # TODO: render
            tracking_results = mpt(self.image_folder)

        # Remove tracklets that are too short
        for person_id in list(tracking_results.keys()):
            if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
                del tracking_results[person_id]

        self.tracking_results = tracking_results
        # logging.info(f'{self.tracking_method} detected {str(len(tracking_results.keys()))} person(s)')

    def _get_pose_generator(self):
        model = PoseGenerator(
            seqlen=16,
            n_layers=2,
            hidden_size=1024,
            pretrained_spin=self.pretrained_spin
        ).to(device)

        model = CheckpointsLoader('checkpoints').load(model, self.pretrained_vibe, strict=False, checkpoints_key='gen_state_dict')
        model.eval()
        return model

    def _run_vibe(self):
        model = self._get_pose_generator()

        for person_id in list(tracking_results.keys()):

            dataset = Inference(
                image_folder=self.image_folder,
                frames=tracking_results[person_id].get('frames'),
                bboxes=tracking_results[person_id].get('bbox'),
                joints2d=tracking_results[person_id].get('joints2d'),
                scale=BBOX_SCALE
            )

            bboxes = dataset.bboxes
            frames = dataset.frames
            joints2d = dataset.joints2d
            has_keypoints = tracking_results[person_id].get('joints2d') is not None

            dataloader = DataLoader(dataset, batch_size=self.vibe_batch_size, num_workers=16)

            with torch.no_grad():

                pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []

                for batch in dataloader:
                    if has_keypoints:
                        batch, nj2d = batch
                        norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                    batch = batch.unsqueeze(0)
                    batch = batch.to(device)

                    batch_size, seqlen = batch.shape[:2]
                    output = model(batch)[-1]

                    pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
                    pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                    pred_pose.append(output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1))
                    pred_betas.append(output['theta'][:, :,75:].reshape(batch_size * seqlen, -1))
                    pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))

                pred_cam = torch.cat(pred_cam, dim=0)
                pred_verts = torch.cat(pred_verts, dim=0)
                pred_pose = torch.cat(pred_pose, dim=0)
                pred_betas = torch.cat(pred_betas, dim=0)
                pred_joints3d = torch.cat(pred_joints3d, dim=0)

                del batch

            # ========= [Optional] run Temporal SMPLify to refine the results ========= #
            if self.run_smplify and self.tracking_method == 'openpose':
                norm_joints2d = np.concatenate(norm_joints2d, axis=0)
                norm_joints2d = convert_kps(norm_joints2d, src='staf', dst='spin')
                norm_joints2d = torch.from_numpy(norm_joints2d).float().to(device)

                # Run Temporal SMPLify
                update, new_opt_vertices, new_opt_cam, new_opt_pose, new_opt_betas, \
                new_opt_joints3d, new_opt_joint_loss, opt_joint_loss = smplify_runner(
                    pred_rotmat=pred_pose,
                    pred_betas=pred_betas,
                    pred_cam=pred_cam,
                    j2d=norm_joints2d,
                    device=device,
                    batch_size=norm_joints2d.shape[0],
                    pose2aa=False,
                )

                # update the parameters after refinement
                # logging.info(f'Update ratio after Temporal SMPLify: {update.sum()} / {norm_joints2d.shape[0]}')
                pred_verts = pred_verts.cpu()
                pred_cam = pred_cam.cpu()
                pred_pose = pred_pose.cpu()
                pred_betas = pred_betas.cpu()
                pred_joints3d = pred_joints3d.cpu()
                pred_verts[update] = new_opt_vertices[update]
                pred_cam[update] = new_opt_cam[update]
                pred_pose[update] = new_opt_pose[update]
                pred_betas[update] = new_opt_betas[update]
                pred_joints3d[update] = new_opt_joints3d[update]

            elif self.run_smplify and self.tracking_method != 'openpose':
                logging.warning('You need to enable pose tracking to run Temporal SMPLify algorithm!')
                logging.warning('Continuing without running Temporal SMPLify...')

            # ========= Save results to a pickle file ========= #
            pred_cam = pred_cam.cpu().numpy()
            pred_verts = pred_verts.cpu().numpy()
            pred_pose = pred_pose.cpu().numpy()
            pred_betas = pred_betas.cpu().numpy()
            pred_joints3d = pred_joints3d.cpu().numpy()

            orig_cam = convert_crop_cam_to_orig_img(
                cam=pred_cam,
                bbox=bboxes,
                img_width=self.orig_width,
                img_height=self.orig_height
            )

            self.results[person_id] = {
                'pred_cam': pred_cam,
                'orig_cam': orig_cam,
                'verts': pred_verts,
                'pose': pred_pose,
                'betas': pred_betas,
                'joints3d': pred_joints3d,
                'joints2d': joints2d,
                'bboxes': bboxes,
                'frame_ids': frames,
            }

        del model

        logging.info(f'Estimated {str(len(self.results.keys()))} 3D poses using VIBE')

    def _clean(self):
        shutil.rmtree(self.image_folder)


class Extractor():

    def __init__(
        self,
        video_folder,
        pretrained_vibe,
        pretrained_spin,
        output_folder='output/',
        tracking_method='yolo',
        run_smplify=False,
        staf_dir='',
        vibe_batch_size=450,
        render=False
    ):
        logging.info('==================== EXTRACTOR ====================')


        self.video_folder = video_folder
        self.pretrained_vibe = pretrained_vibe
        self.pretrained_spin = pretrained_spin

        self.output_folder = output_folder
        self.tracking_method = tracking_method
        self.run_smplify = run_smplify
        self.staf_dir = staf_dir
        self.vibe_batch_size = vibe_batch_size
        self.render = render

    def run(self):
        """Main function to execute the extractor"""

        for video_file in self._video_files():
            
            # Check if video file exists
            if not os.path.isfile(video_file):
                logging.error(f'Skipping video \"{video_file}\": does not exist!')
                continue

            # Extract from single video
            sve = SingleVideoExtractor(video_file, pretrained_spin=self.pretrained_spin)
            sve.run(render=self.render, output_folder=self.output_folder)


    def _video_files(self):
        video_files = glob.glob(os.path.join(self.video_folder, '*'))
        return video_files
