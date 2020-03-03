import argparse
import logging
import os

from extractor import Extractor
from synthesiser import Synthesiser
from utils import make_tarfile

def main(args):
    if not args.skip_extractor:
        extractor = Extractor(video_folder=args.video_folder,
                              output_folder=args.extractor_results_folder,
                              render=args.render_extractor_results,
                              tracking_method=args.tracking_method,
                              staf_dir=args.staf_dir,
                              run_smplify=args.run_smplify)
        extractor.run()

    synthesiser = Synthesiser(blender=args.blender,
                              debug_blender=args.debug_blender,
                              motion_path=args.extractor_results_folder,
                              target_size=args.target_size,
                              num_frames=args.num_frames)
    synthesiser.run()

    make_tarfile(args.output, 'output')

    logging.info('Done.')


if __name__ == '__main__':
    logging.basicConfig(level='INFO', format='%(message)s')

    parser = argparse.ArgumentParser()

    parser.add_argument('--video_folder',
                        type=str,
                        help='input videos directory path')

    parser.add_argument('--extractor_results_folder',
                        type=str,
                        default='data/motion',
                        help='Folder to save extractor results')

    parser.add_argument('--render_extractor_results',
                        action='store_true',
                        help='Render the results of the motion extractor into videos')

    parser.add_argument('--skip_extractor',
                        action='store_true',
                        help='SKip the motion extraction phase')

    parser.add_argument('--blender',
                        type=str,
                        default='blender/blender',
                        help='Path to Blender executable')

    parser.add_argument('--target_size',
                        type=int,
                        default=100,
                        help='Target size of the synthetic dataset (number of videos)')

    parser.add_argument('--output',
                        type=str,
                        default='dataset',
                        help='Where to write the output archive to')

    parser.add_argument('--num_frames',
                        type=int,
                        default=200,
                        help='Maximum number of frames for each synthetic video')

    parser.add_argument('--debug_blender',
                        action='store_true',
                        help='Open Blender with the first scene, for debugging purposes.')


    parser.add_argument('--staf_dir',
                        type=str,
                        default='',
                        help='Path to the STAF build of Openpose to use when tracking_method is pose.')

    parser.add_argument('--tracking_method',
                        type=str,
                        default='bbox',
                        help='bbox or pose')

    parser.add_argument('--run_smplify',
                        action='store_true',
                        help='')

    args = parser.parse_args()

    if not args.skip_extractor and not args.video_folder:
        raise Exception('--video_folder is required for the extractor.')

    main(args)
