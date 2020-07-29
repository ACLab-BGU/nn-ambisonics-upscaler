import sys

from src.models import prepare_model
import src.options as options
from src.utils import prepare_logger, prepare_trainer


def train(opts):
    base_opts = options.get_default_opts(opts)
    full_opts = options.prepare_opts(base_opts,opts)
    full_opts = options.validate_opts(full_opts)

    model = prepare_model(full_opts)
    logger = prepare_logger(full_opts)
    trainer = prepare_trainer(full_opts,logger)

    trainer.fit(model)


if __name__ == '__main__':
    assert len(sys.argv) > 1, "Please insert a config file to run"
    opts = sys.argv[1]
    train(opts)
