"""Training loop for segmentation (stub)."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml')
    args = parser.parse_args()
    print('Training segmentation with config', args.config)


if __name__ == '__main__':
    main()
