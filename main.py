import argparse

from extractor import Extractor
from synthesiser import Synthesiser

def main(args):
  extractor = Extractor(video_folder=args.video_folder, output_folder=args.output_folder)
  extractor.run()

  synthesiser = Synthesiser()
  synthesiser.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_folder',
                        type=str,
                        help='input videos directory path',
                        required=True)

    parser.add_argument('--output_folder',
                        type=str,
                        default='output/',
                        help='output folder to write results')

    args = parser.parse_args()

    main(args)
