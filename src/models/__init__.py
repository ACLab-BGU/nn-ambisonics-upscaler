from src.models import fc_model, cnn_model, rnn_model


def find_model_using_name(model_name):
    if model_name == 'fc':
        model = fc_model.FC
        default_opts = fc_model.default_opts
    elif model_name == 'cnn':
        model = cnn_model.CNN
        default_opts = cnn_model.default_opts
    elif model_name == 'rnn_NNIWF':
        model = rnn_model.NNIWF
        default_opts = rnn_model.default_opts
    else:
        raise NotImplementedError

    return model, default_opts


def prepare_model(opts):
    model, _ = find_model_using_name(opts['model_name'])
    instance = model(opts)
    return instance
