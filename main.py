import argparse

from extractor import Extractor
from synthesiser import Synthesiser

def main(args):
  extractor = Extractor(video_folder=args.video_folder,
                        output_folder=args.extractor_results_folder,
                        render=args.render_extractor_results)
  extractor.run()

  synthesiser = Synthesiser(blender=args.blender,
                            motion_path=args.extractor_results_folder,
                            target_size=args.target_size,
                            num_frames=args.num_frames)
  synthesiser.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_folder',
                        type=str,
                        help='input videos directory path',
                        required=True)

    parser.add_argument('--extractor_results_folder',
                        type=str,
                        default='data/motion',
                        help='Folder to save extractor results')

    parser.add_argument('--render_extractor_results',
                        action='store_true',
                        help='Render the results of the motion extractor into videos')

    parser.add_argument('--blender',
                        type=str,
                        default='./blender/blender',
                        help='Path to Blender executable')

    parser.add_argument('--target_size',
                        type=int,
                        default=100,
                        help='Target size of the synthetic dataset (number of videos)')

    parser.add_argument('--num_frames',
                        type=int,
                        default=200,
                        help='Maximum number of frames for each synthetic video')

    args = parser.parse_args()

    main(args)
