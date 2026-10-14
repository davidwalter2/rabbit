from rabbit import common

# dictionary with class name and the corresponding filename where it is defined
baseline_constraints = {
    "Bounds": "constraints",
}


def load_constraints(class_name, indata, *args, **kwargs):
    constraints = common.load_class_from_module(
        class_name, baseline_constraints, base_dir="rabbit.constraints"
    )
    return constraints.parse_args(indata, *args, **kwargs)
