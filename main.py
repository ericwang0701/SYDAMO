import argparse

from extractor import Extractor
from synthesiser import Synthesiser

def main(args):
  extractor = Extractor(video_folder=args.video_folder,
                        output_folder=args.output_folder,
                        render=args.render_extractor_results)
  extractor.run()

  synthesiser = Synthesiser(blender=args.blender)
  synthesiser.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_folder',
                        type=str,
                        help='input videos directory path',
                        required=True)

    parser.add_argument('--output_folder',
                        type=str,
                        default='data/motion',
                        help='output folder to write results')

    parser.add_argument('--render_extractor_results',
                        action='store_true',
                        help='Render the results of the motion extractor into videos')

    parser.add_argument('--blender',
                        type=str,
                        default='./blender/blender',
                        help='Path to Blender executable')

    args = parser.parse_args()

    main(args)
