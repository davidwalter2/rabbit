from rabbit import common

# dictionary with class name and the corresponding filename where it is defined
baseline_models = {
    "Ones": "poi_model",
    "Mu": "poi_model",
    "MixtureModel": "poi_model",
}


def load_model(class_name, indata, **kwargs):
    model_class = common.load_class_from_module(
        class_name, baseline_models, base_dir="rabbit.poi_models"
    )
    model_instance = model_class(indata, **kwargs)
    return model_instance
