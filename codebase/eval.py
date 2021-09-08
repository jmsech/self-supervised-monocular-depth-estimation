from depth_model import *
from Dataloader import Kittiloader
from dataset import DataGenerator
from evaluate_model import evaluate
import argparse
import pathlib
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Evaluation options.')

parser.add_argument('--model_path', type=pathlib.Path,
                    help='path to load the model from')
parser.add_argument('--kitti_dir', type=pathlib.Path)

args = parser.parse_args()


def main():
    batch_size = 1

    model = torch.load(args.model_path).cuda()
    datagen_test = DataGenerator(args.kitti_dir, phase='test', splits='eigen')
    test_dataloader = datagen_test.create_data(batch_size)
    results = evaluate(model, test_dataloader, display=False)
    print(results)

if __name__ == '__main__':
    main()