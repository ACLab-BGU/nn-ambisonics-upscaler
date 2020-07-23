import sys

from src.models import prepare_model
import src.options as options
from src.utils import prepare_logger, prepare_trainer


def train(opts):
    base_opts = options.get_default_opts(opts)
    opts = options.prepare_opts(base_opts,opts)
    opts = options.validate_opts(opts)

    model = prepare_model(opts)
    logger = prepare_logger(opts)
    trainer = prepare_trainer(opts,logger)

    trainer.fit(model)


if __name__ == '__main__':
    opts = sys.argv[1]
    train(opts)
