import src.models.fc_model as fc_model


def find_model_using_name(model_name):
    if model_name == 'fc':
        model = fc_model.BaseModelLT
        default_opts = fc_model.default_opts
    else:
        raise NotImplementedError

    return model, default_opts


def prepare_model(opts):
    model, _ = find_model_using_name(opts['model_name'])
    instance = model(opts)
    return instance
