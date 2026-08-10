"""
IsoMT convenience wrappers for the four ABCD param models.

Hardcode the axis conventions used in CMS WMass nonprompt background
estimation:
  - y-axis: "relIso"  (bin 0 = pass/signal, bin 1 = fail/sideband)
  - x-axis: "mt"      (last bin = signal, earlier bins = sidebands)
  - smoothing axis: "pt"  (for the smooth variants only)

All four wrappers accept a single channel name and derive the per-region
channel dicts automatically. The extended variants require at least 3 mt bins;
the plain variants require at least 2.

CLI syntax:
    --paramModel ABCDIsoMT               [yieldCorrection:0|1] <process> <channel>
    --paramModel ExtendedABCDIsoMT       [yieldCorrection:0|1] <process> <channel>
    --paramModel SmoothABCDIsoMT         [order:N] [yieldCorrection:0|1] <process> <channel>
    --paramModel SmoothExtendedABCDIsoMT [params:<src> | order:N] \\
                                         [yieldCorrection:0|1] <process> <channel>

``yieldCorrection`` defaults to 0 (no MC factor); pass ``yieldCorrection:1``
to enable the yield-level form. See the underlying ABCD model docstrings.

Region layout (shared by all four):

             relIso=signal(0)   relIso=sideband(1)
mt=extra-sb:       Bx                  Ax
mt=sideband:       B                   A
mt=signal:         D  ← predicted      C
"""

from wums import logging

from rabbit.auxiliary import initial_params_name, read_initial_params
from rabbit.param_models.abcd_model import ABCD
from rabbit.param_models.extended_abcd_model import ExtendedABCD
from rabbit.param_models.smooth_abcd_model import SmoothABCD
from rabbit.param_models.smooth_extended_abcd_model import (
    SmoothExtendedABCD,
    resolve_params_token,
)

logger = logging.child_logger(__name__)


def _isomtmt_regions(indata, channel_name):
    """Return (mt_signal_idx, mt_sideband_idx, mt_extra_sideband_idx) from the channel's mt axis size."""
    axes = indata.channel_info[channel_name]["axes"]
    mt_ax = next((a for a in axes if a.name == "mt"), None)
    if mt_ax is None:
        raise ValueError(
            f"Channel '{channel_name}' has no axis named 'mt'. "
            f"Available axes: {[a.name for a in axes]}"
        )
    if next((a for a in axes if a.name == "relIso"), None) is None:
        raise ValueError(
            f"Channel '{channel_name}' has no axis named 'relIso'. "
            f"Available axes: {[a.name for a in axes]}"
        )
    return mt_ax.size - 1, mt_ax.size - 2, mt_ax.size - 3


def _pop_yield_correction(tokens):
    """Consume an optional ``yieldCorrection:N`` token from the front of the list."""
    if tokens and tokens[0].startswith("yieldCorrection:"):
        return bool(int(tokens.pop(0).split(":", 1)[1]))
    return False


def _parse_isomtmt_args(tokens):
    """Parse [order:N] [yieldCorrection:0|1] <process> <channel> from a token list."""
    order = 1
    if tokens and tokens[0].startswith("order:"):
        order = int(tokens.pop(0).split(":", 1)[1])
    yield_correction = _pop_yield_correction(tokens)
    if len(tokens) < 2:
        raise ValueError("Expected <process> <channel>")
    process = tokens.pop(0)
    channel = tokens.pop(0)
    if tokens:
        raise ValueError(f"Unexpected extra arguments: {tokens}")
    return order, process, channel, yield_correction


def _parse_isomtmt_args_with_params(tokens, indata, model_name):
    """Parse [params:<src> | order:N] [yieldCorrection:0|1] <process> <channel>
    from a token list.

    ``params:`` and ``order:`` are mutually exclusive; the params source encodes
    the polynomial order via its stored ``order`` field.

    With neither token, the initial parameters are taken from the auxiliary bundle
    ``initial_params_<model>_<process>_<channel>`` of the input file if the tool
    that wrote the datacard stored one, and are left at zero otherwise.
    """
    params_token = None
    order = None
    if tokens and tokens[0].startswith("params:"):
        params_token = tokens.pop(0)
    elif tokens and tokens[0].startswith("order:"):
        order = int(tokens.pop(0).split(":", 1)[1])
    yield_correction = _pop_yield_correction(tokens)
    if len(tokens) < 2:
        raise ValueError("Expected <process> <channel>")
    process = tokens.pop(0)
    channel = tokens.pop(0)
    if tokens:
        raise ValueError(f"Unexpected extra arguments: {tokens}")

    initial_params = None
    if params_token is not None:
        initial_params, order = resolve_params_token(params_token, indata)
    elif order is None:
        # no explicit request: pick up values shipped with the input file, if any
        aux_name = initial_params_name(model_name, process, channel)
        initial_params, order = read_initial_params(indata, aux_name)
        if initial_params is not None:
            logger.info(
                f"Using initial parameters from auxiliary bundle '{aux_name}' "
                f"of the input file (order {order})"
            )

    return (
        order if order is not None else 1,
        process,
        channel,
        initial_params,
        yield_correction,
    )


class ABCDIsoMT(ABCD):
    """
    ABCD in the (mt × relIso) plane for a single channel.

    Uses the last two mt bins as sideband / signal and relIso bins 0 / 1 as
    signal / sideband.  All other axes in the channel become free per-bin
    parameters (outer axes of the ABCD model).
    """

    def __init__(self, indata, abcd_process, channel_name, **kwargs):
        mt_s, mt_sb, _ = _isomtmt_regions(indata, channel_name)
        if mt_sb < 0:
            raise ValueError(f"Channel '{channel_name}' mt axis has fewer than 2 bins")
        channel_A = {channel_name: {"relIso": 1, "mt": mt_sb}}
        channel_B = {channel_name: {"relIso": 0, "mt": mt_sb}}
        channel_C = {channel_name: {"relIso": 1, "mt": mt_s}}
        channel_D = {channel_name: {"relIso": 0, "mt": mt_s}}
        super().__init__(
            indata, abcd_process, channel_A, channel_B, channel_C, channel_D, **kwargs
        )

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        tokens = list(args)
        yield_correction = _pop_yield_correction(tokens)
        if len(tokens) < 2:
            raise ValueError(
                "ABCDIsoMT expects: [yieldCorrection:0|1] <process> <channel>"
            )
        process, channel = tokens[0], tokens[1]
        if len(tokens) > 2:
            raise ValueError(f"Unexpected extra arguments: {tokens[2:]}")
        return cls(
            indata, process, channel, yield_correction=yield_correction, **kwargs
        )


class ExtendedABCDIsoMT(ExtendedABCD):
    """
    ExtendedABCD in the (mt × relIso) plane for a single channel.

    Uses the last three mt bins as extra-sideband / sideband / signal and
    relIso bins 0 / 1 as signal / sideband.
    """

    def __init__(self, indata, abcd_process, channel_name, **kwargs):
        mt_s, mt_sb, mt_xsb = _isomtmt_regions(indata, channel_name)
        if mt_xsb < 0:
            raise ValueError(f"Channel '{channel_name}' mt axis has fewer than 3 bins")
        channel_Ax = {channel_name: {"relIso": 1, "mt": mt_xsb}}
        channel_Bx = {channel_name: {"relIso": 0, "mt": mt_xsb}}
        channel_A = {channel_name: {"relIso": 1, "mt": mt_sb}}
        channel_B = {channel_name: {"relIso": 0, "mt": mt_sb}}
        channel_C = {channel_name: {"relIso": 1, "mt": mt_s}}
        channel_D = {channel_name: {"relIso": 0, "mt": mt_s}}
        super().__init__(
            indata,
            abcd_process,
            channel_A,
            channel_B,
            channel_C,
            channel_D,
            channel_Ax,
            channel_Bx,
            **kwargs,
        )

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        tokens = list(args)
        yield_correction = _pop_yield_correction(tokens)
        if len(tokens) < 2:
            raise ValueError(
                "ExtendedABCDIsoMT expects: [yieldCorrection:0|1] <process> <channel>"
            )
        process, channel = tokens[0], tokens[1]
        if len(tokens) > 2:
            raise ValueError(f"Unexpected extra arguments: {tokens[2:]}")
        return cls(
            indata, process, channel, yield_correction=yield_correction, **kwargs
        )


class SmoothABCDIsoMT(SmoothABCD):
    """
    SmoothABCD in the (mt × relIso) plane, smoothed along the pt axis.

    Uses the last two mt bins and relIso bins 0 / 1.  The pt axis is the
    smoothing axis; all remaining axes become outer axes.
    """

    def __init__(self, indata, abcd_process, channel_name, order=1, **kwargs):
        mt_s, mt_sb, _ = _isomtmt_regions(indata, channel_name)
        if mt_sb < 0:
            raise ValueError(f"Channel '{channel_name}' mt axis has fewer than 2 bins")
        channel_A = {channel_name: {"relIso": 1, "mt": mt_sb}}
        channel_B = {channel_name: {"relIso": 0, "mt": mt_sb}}
        channel_C = {channel_name: {"relIso": 1, "mt": mt_s}}
        channel_D = {channel_name: {"relIso": 0, "mt": mt_s}}
        super().__init__(
            indata,
            "pt",
            abcd_process,
            channel_A,
            channel_B,
            channel_C,
            channel_D,
            order=order,
            **kwargs,
        )

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        tokens = list(args)
        order, process, channel, yield_correction = _parse_isomtmt_args(tokens)
        return cls(
            indata,
            process,
            channel,
            order=order,
            yield_correction=yield_correction,
            **kwargs,
        )


class SmoothExtendedABCDIsoMT(SmoothExtendedABCD):
    """
    SmoothExtendedABCD in the (mt × relIso) plane, smoothed along the pt axis.

    Uses the last three mt bins and relIso bins 0 / 1.  The pt axis is the
    smoothing axis; all remaining axes become outer axes.

    CLI syntax:
        --paramModel SmoothExtendedABCDIsoMT [params:<src> | order:N] <process> <channel>

    ``params:<src>`` loads initial parameter values and the polynomial order,
    either from ``params:aux:<name>`` (an auxiliary bundle in the input file) or
    from ``params:<file.hdf5>`` (a standalone file, e.g. from WRemnants'
    regen_smoothing_params.py).  It is mutually exclusive with ``order:N``.

    With neither token given, the initial parameters are read from the auxiliary
    bundle ``initial_params_SmoothExtendedABCDIsoMT_<process>_<channel>`` if the
    input file carries one (WRemnants' setupRabbit.py writes it for the
    simultaneous ABCD setup), and are left at zero otherwise.
    """

    def __init__(self, indata, abcd_process, channel_name, order=1, **kwargs):
        mt_s, mt_sb, mt_xsb = _isomtmt_regions(indata, channel_name)
        if mt_xsb < 0:
            raise ValueError(f"Channel '{channel_name}' mt axis has fewer than 3 bins")
        channel_Ax = {channel_name: {"relIso": 1, "mt": mt_xsb}}
        channel_Bx = {channel_name: {"relIso": 0, "mt": mt_xsb}}
        channel_A = {channel_name: {"relIso": 1, "mt": mt_sb}}
        channel_B = {channel_name: {"relIso": 0, "mt": mt_sb}}
        channel_C = {channel_name: {"relIso": 1, "mt": mt_s}}
        channel_D = {channel_name: {"relIso": 0, "mt": mt_s}}
        super().__init__(
            indata,
            "pt",
            abcd_process,
            channel_A,
            channel_B,
            channel_C,
            channel_D,
            channel_Ax,
            channel_Bx,
            order=order,
            **kwargs,
        )

    @classmethod
    def parse_args(cls, indata, *args, **kwargs):
        tokens = list(args)
        (
            order,
            process,
            channel,
            initial_params,
            yield_correction,
        ) = _parse_isomtmt_args_with_params(tokens, indata, cls.__name__)
        return cls(
            indata,
            process,
            channel,
            order=order,
            initial_params=initial_params,
            yield_correction=yield_correction,
            **kwargs,
        )
