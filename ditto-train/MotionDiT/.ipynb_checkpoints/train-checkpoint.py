import tyro
import traceback
from src.options.option import TrainOptions, check_train_opt
from src.trainers.trainer import Trainer


def main():
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")

    opt = tyro.cli(TrainOptions)
    print(opt)

    check_train_opt(opt)

    trainer = Trainer(opt)
    trainer.train_loop()


if __name__ == '__main__':
    main()