import argparse


class Trainer:
    def __init__(self):
        self.args = None

        self.parse_args()
        self.init_dataset()
        self.init_model()
        self.train()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        # Training Setting
        parser.add_argument('--epoch-start', type=int, default=0)
        parser.add_argument('--epoch-end', type=int, default=250000)
        # Checkpoint
        parser.add_argument('--checkpoint', type=bool, action=argparse.BooleanOptionalAction)
        parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
        parser.add_argument('--checkpoint-save-interval', type=int, default=1000)

        self.args = parser.parse_args()

    def init_dataset(self):
        pass

    def init_model(self):
        pass

    def train(self):
        pass

    @staticmethod
    def log(message):
        print(f'{message}')


if __name__ == '__main__':
    Trainer()
