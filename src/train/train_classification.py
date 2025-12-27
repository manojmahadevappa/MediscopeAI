"""Training loop for classification (stub)."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml')
    args = parser.parse_args()
    print('Training classification with config', args.config)


if __name__ == '__main__':
    main()
