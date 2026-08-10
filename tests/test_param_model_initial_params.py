"""Test initial parameters for SmoothExtendedABCDIsoMT shipped with the datacard.

The tool writing the datacard can store starting values for the param model in an
``auxiliary`` bundle (see :mod:`rabbit.auxiliary`), so they stay consistent with
the templates they were derived from and no separate file has to be passed on the
command line.

Covers:
  * automatic pickup of ``initial_params_<model>_<process>_<channel>`` when no
    ``params:``/``order:`` token is given;
  * the explicit ``params:aux:<name>`` token;
  * the explicit ``params:<file.hdf5>`` token (standalone file, backwards compatible);
  * ``order:N`` disables the automatic pickup and starts from zero;
  * no bundle present -> zero starting values;
  * a mistyped bundle name raises.
"""

import os
import tempfile

import h5py
import hist
import numpy as np

from rabbit import inputdata, tensorwriter
from rabbit.auxiliary import initial_params_name
from rabbit.param_models.helpers import load_models

MODEL = "SmoothExtendedABCDIsoMT"
PROCESS = "Fake"
CHANNEL = "ch0"
ORDER = 2


def build_isomt_writer():
    """Minimal datacard with the (pt x relIso x mt) layout the IsoMT models expect."""
    rng = np.random.default_rng(7)
    axes = [
        hist.axis.Regular(3, 0, 3, name="eta", underflow=False, overflow=False),
        hist.axis.Regular(5, 26, 56, name="pt", underflow=False, overflow=False),
        hist.axis.Variable([0, 0.15, np.inf], name="relIso", overflow=False),
        hist.axis.Variable([0, 20, 40, np.inf], name="mt", overflow=False),
    ]
    shape = tuple(a.size for a in axes)

    h_fake = hist.Hist(*axes, storage=hist.storage.Weight())
    h_fake.values()[...] = rng.uniform(10.0, 100.0, shape)
    h_fake.variances()[...] = h_fake.values()

    h_sig = hist.Hist(*axes, storage=hist.storage.Weight())
    h_sig.values()[...] = rng.uniform(1.0, 10.0, shape)
    h_sig.variances()[...] = h_sig.values()

    h_data = hist.Hist(*axes, storage=hist.storage.Double())
    h_data.view()[...] = rng.poisson(h_fake.values() + h_sig.values())

    writer = tensorwriter.TensorWriter()
    writer.add_channel(axes, CHANNEL)
    writer.add_data(h_data, CHANNEL)
    writer.add_process(h_sig, "sig", CHANNEL, signal=True)
    writer.add_process(h_fake, PROCESS, CHANNEL, signal=False)
    return writer


def write_datacard(tmpdir, name, datasets=None):
    writer = build_isomt_writer()
    if datasets is not None:
        writer.add_auxiliary(initial_params_name(MODEL, PROCESS, CHANNEL), datasets)
    writer.write(outfolder=tmpdir, outfilename=name)
    return inputdata.FitInputData(os.path.join(tmpdir, f"{name}.hdf5"))


def build_model(indata, *tokens):
    return load_models([[MODEL, *tokens, PROCESS, CHANNEL]], indata)


def test_initial_params_from_auxiliary(tmpdir):
    # nparams depends only on the channel layout and the order, so ask the model
    # itself rather than hardcoding the region/outer-bin bookkeeping here
    indata_plain = write_datacard(tmpdir, "no_aux")
    nparams = build_model(indata_plain, f"order:{ORDER}").nparams
    assert np.all(
        build_model(indata_plain, f"order:{ORDER}").xparamdefault.numpy() == 0
    ), "expected zero starting values without an auxiliary bundle"

    params = np.linspace(-0.5, 0.5, nparams).astype(np.float64)
    datasets = {"params": params, "order": np.array(ORDER)}
    indata_obj = write_datacard(tmpdir, "with_aux", datasets)

    model = build_model(indata_obj)
    assert model.order == ORDER, f"order {model.order} != {ORDER}"
    assert model.nparams == nparams
    assert np.allclose(
        model.xparamdefault.numpy(), params.astype(np.float32)
    ), "automatic pickup did not use the auxiliary bundle"
    print("PASS: initial parameters picked up automatically from the auxiliary bundle")

    aux_name = initial_params_name(MODEL, PROCESS, CHANNEL)
    explicit = build_model(indata_obj, f"params:aux:{aux_name}")
    assert np.array_equal(
        explicit.xparamdefault.numpy(), model.xparamdefault.numpy()
    ), "explicit params:aux: token differs from the automatic pickup"
    print("PASS: explicit params:aux:<name> token")

    # explicit order must win over whatever the file carries, starting from zero
    other_order = build_model(indata_obj, "order:1")
    assert other_order.order == 1
    assert np.all(
        other_order.xparamdefault.numpy() == 0
    ), "order:N should not pick up the auxiliary bundle"
    print("PASS: order:N disables the automatic pickup")

    try:
        build_model(indata_obj, "params:aux:does_not_exist")
    except ValueError as exc:
        assert "does_not_exist" in str(exc)
        print("PASS: unknown auxiliary bundle name raises")
    else:
        raise AssertionError("expected ValueError for an unknown auxiliary bundle")


def test_initial_params_from_file(tmpdir):
    """The standalone params file keeps working alongside the auxiliary bundle."""
    indata_obj = write_datacard(tmpdir, "for_file")
    nparams = build_model(indata_obj, f"order:{ORDER}").nparams
    params = np.linspace(1.0, 2.0, nparams).astype(np.float64)

    path = os.path.join(tmpdir, "params.hdf5")
    with h5py.File(path, "w") as f:
        f.create_dataset("params", data=params)
        f.create_dataset("order", data=np.array(ORDER))

    model = build_model(indata_obj, f"params:{path}")
    assert model.order == ORDER
    assert np.allclose(model.xparamdefault.numpy(), params.astype(np.float32))
    print("PASS: standalone params:<file.hdf5> token")


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_initial_params_from_auxiliary(tmpdir)
        test_initial_params_from_file(tmpdir)
    print()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
