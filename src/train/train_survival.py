"""Training loop for survival modeling (stub)."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml')
    args = parser.parse_args()
    print('Training survival model with config', args.config)
    print('Activating virtual environment and starting the web app...')



if __name__ == '__main__':
    main()
