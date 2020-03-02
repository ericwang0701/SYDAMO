import argparse
import logging
import tarfile
import os

from extractor import Extractor
from synthesiser import Synthesiser


def main(args):
    if not args.skip_extractor:
        extractor = Extractor(video_folder=args.video_folder,
                              output_folder=args.extractor_results_folder,
                              render=args.render_extractor_results)
        extractor.run()

    synthesiser = Synthesiser(blender=args.blender,
                              motion_path=args.extractor_results_folder,
                              target_size=args.target_size,
                              num_frames=args.num_frames)
    synthesiser.run()

    make_tarfile(args.output, 'output')


def make_tarfile(archive_name, video_folder):
    with tarfile.open(f'{archive_name}.tar.gz', 'w:gz') as tar:
        tar.add(video_folder, arcname=os.path.basename(video_folder))


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
                        help='Name of the output archive')

    parser.add_argument('--num_frames',
                        type=int,
                        default=200,
                        help='Maximum number of frames for each synthetic video')

    # parser.add_argument('--archive',
    #                     action='store_true',
    #                     help='Create an archive with the whole synthetic dataset')

    args = parser.parse_args()

    if not args.skip_extractor and not args.video_folder:
        raise Exception('--video_folder is required for the extractor.')

    main(args)
