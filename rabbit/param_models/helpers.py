from rabbit import common

# dictionary with class name and the corresponding filename where it is defined
baseline_models = {
    "Ones": "param_model",
    "Mu": "param_model",
    "Mixture": "param_model",
    "ABCD": "abcd_model",
}


def load_model(class_name, indata, *args, **kwargs):
    model = common.load_class_from_module(
        class_name, baseline_models, base_dir="rabbit.param_models"
    )
    return model.parse_args(indata, *args, **kwargs)
