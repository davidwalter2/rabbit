#!/usr/bin/env python3

import copy

import tensorflow as tf

tf.config.experimental.enable_op_determinism()

import time

import numpy as np
from scipy.stats import norm

from rabbit import asymptotic_limits, fitter, inputdata, parsing, workspace
from rabbit.mappings import helpers as mh
from rabbit.mappings import mapping as mp
from rabbit.poi_models import helpers as ph
from rabbit.tfhelpers import edmval_cov

from wums import output_tools, logging  # isort: skip

logger = None


def make_parser():
    parser = parsing.common_parser()

    parser.add_argument(
        "--asymptoticLimits",
        nargs="+",
        default=[],
        type=str,
        help="Compute asymptotic upper limit based on CLs based on specified parameter, either on the parameter itself or on the (masked) channel via '--limitsOnChannel'",
    )
    parser.add_argument(
        "--limitsOnChannel",
        nargs="+",
        default=[],
        type=str,
        help="Compute asymptotic upper limit based on (masked) channel",
    )
    parser.add_argument(
        "--cls",
        nargs="+",
        default=[0.05],
        type=float,
        help="Confidence level for asymptotic upper limit, multiple are possible",
    )

    return parser.parse_args()


def do_asymptotic_limits(
    args, fitter, ws, data_values, bkg_only_fit=False, on_channel=[]
):
    if bkg_only_fit:
        logger.info("Perform background only fit")
        # set process to zero and freeze
        fitter.freeze_params(args.asymptoticLimits)

        fitter.minimize()

        # set asimov from background only fit
        fitter.set_nobs(fitter.expected_yield())

        # defreeze process again to evaluate it's dependencies
        fitter.defreeze_params(args.asymptoticLimits)

    if not args.noHessian:
        # compute the covariance matrix and estimated distance to minimum

        val, grad, hess = fitter.loss_val_grad_hess()
        edmval, cov = edmval_cov(grad, hess)

        logger.info(f"edmval: {edmval}")

        fitter.cov.assign(cov)

        del cov

    # Clone fitter to simultaneously minimize on asimov dataset
    fitter_asimov = copy.deepcopy(fitter)

    # unconditional fit to real data
    fitter.set_nobs(data_values)
    fitter.minimize()

    # asymptotic limits (CLs)
    #  see:
    #  - combine tutorial https://indico.cern.ch/event/976099/contributions/4138520/
    #  - paper: https://arxiv.org/abs/1007.1727

    clb_list = np.array([0.025, 0.16, 0.5, 0.84, 0.975])

    # axes for output histogram: params, cls, clb
    limits_shape = [len(args.asymptoticLimits), len(args.cls), len(clb_list)]
    limits_obs_shape = [len(args.asymptoticLimits), len(args.cls)]

    limits = np.full(limits_shape, np.nan)
    limits_obs = np.full(limits_obs_shape, np.nan)
    limits_nll = np.full(limits_shape, np.nan)
    limits_nll_obs = np.full(limits_obs_shape, np.nan)
    for i, key in enumerate(args.asymptoticLimits):

        if len(on_channel):
            channel = on_channel[i]
            if channel not in fitter_asimov.indata.channel_info.keys():
                raise ValueError(
                    f"Can not find (masked) channel {channel} to set limits"
                )

            logger.info(f"Get the limit on the masked channel {channel}")
            mapping = mh.load_mapping("Select", fitter_asimov.indata, channel)
            fun = mapping.compute_flat

            exp, exp_var, _0, _1, _2 = fitter_asimov.expected_with_variance(
                fun,
                profile=True,
                compute_cov=False,
                compute_global_impacts=False,
                need_observables=True,
                inclusive=True,
            )
            xbest = exp.numpy()[0]
            xerr = exp_var.numpy()[0] ** 0.5

            # for observed limit
            exp, exp_var, _0, _1, _2 = fitter.expected_with_variance(
                fun,
                profile=True,
                compute_cov=False,
                compute_global_impacts=False,
                need_observables=True,
                inclusive=True,
            )
            xobs = exp.numpy()[0]
            xobs_err = exp_var.numpy()[0] ** 0.5
        else:
            if key not in fitter.poi_model.pois.astype(str):
                raise RuntimeError(
                    f"Can not compute asymptotic limits for parameter {key}, no such signal process found, signal processe are: {fitter.poi_model.pois.astype(str)}"
                )
            logger.info(f"Get the limit from the signal strength")
            idx = np.where(fitter.parms.astype(str) == key)[0][0]
            xbest = fitter_asimov.get_blinded_poi()[idx]
            xerr = fitter_asimov.cov[idx, idx] ** 0.5

            xbest = xbest.numpy()
            xerr = xerr.numpy()

            # for observed limit
            xobs = fitter.get_blinded_poi()[idx]
            val, grad, hess = fitter.loss_val_grad_hess()
            edmval, cov = edmval_cov(grad, hess)
            fitter.cov.assign(cov)
            xobs_err = fitter.cov[idx, idx] ** 0.5

            if not args.allowNegativePOI:
                xerr = 2 * xerr * xbest**0.5
                xobs_err = 2 * xobs_err * xobs**0.5

        logger.debug(f"Best fit {key} = {xbest} +/- {xerr}")

        # this is the denominator of q for likelihood based limits
        nllvalreduced_asimov = fitter_asimov.reduced_nll().numpy()

        for j, cl_s in enumerate(args.cls):
            logger.info(f" -- AsymptoticLimits ( CLs={round(cl_s*100,1):4.1f}% ) -- ")

            # now we need to find the values for mu where q_{mu,A} = -2ln(L)
            for k, cl_b in enumerate(clb_list):
                cl_sb = cl_s * cl_b
                qmuA = (norm.ppf(cl_b) - norm.ppf(cl_sb)) ** 2
                logger.debug(f"Find r with q_(r,A)=-2ln(L)/ln(L0) = {qmuA}")

                # Gaussian approximation
                r = xbest + xerr * qmuA**0.5
                logger.info(
                    f"Expected (Gaussian) {round((cl_b)*100,1):4.1f}%: {key} < {r}"
                )
                limits[i, j, k] = r

                if len(on_channel):
                    # TODO: implement for channels
                    pass
                else:
                    # Likelihood approximation
                    r = fitter_asimov.contour_scan(
                        key, nllvalreduced_asimov, qmuA, signs=[1]
                    )[0]
                    logger.info(
                        f"Expected (Likelihood) {round((cl_b)*100,1):4.1f}%: {key} < {r}"
                    )
                    limits_nll[i, j, k] = r

            # Gaussian approximation
            limits_obs[i, j] = asymptotic_limits.compute_gaussian_limit(
                key, xobs, xobs_err, xerr, cl_s
            )

            # Likelihood approximation
            if len(on_channel):
                # TODO: implement for channels
                pass
            else:
                # TODO: make it work
                # nllvalreduced = fitter.reduced_nll().numpy()
                # limits_nll_obs[i, j] = asymptotic_limits.compute_likelihood_limit(fitter, fitter_asimov, nllvalreduced, nllvalreduced_asimov, key, cl_s)
                pass

    ws.add_limits_hist(
        limits,
        args.asymptoticLimits,
        args.cls,
        clb_list,
        base_name="gaussian_asymptoticLimits_expected",
    )

    ws.add_limits_hist(
        limits_obs,
        args.asymptoticLimits,
        args.cls,
        base_name="gaussian_asymptoticLimits_observed",
    )

    if len(on_channel):
        # TODO: implement for channels
        ws.add_limits_hist(
            limits_nll,
            args.asymptoticLimits,
            args.cls,
            clb_list,
            base_name="likelihood_asymptoticLimits_expected",
        )

        # TODO: make it work
        # ws.add_limits_hist(
        #     limits_nll_obs,
        #     args.asymptoticLimits,
        #     args.cls,
        #     base_name="likelihood_asymptoticLimits_observed",
        # )


def main():
    start_time = time.time()
    args = make_parser()

    if args.eager:
        tf.config.run_functions_eagerly(True)

    global logger
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    # make list of fits with -1: asimov; 0: fit to data; >=1: toy
    fits = np.concatenate(
        [np.array([x]) if x <= 0 else 1 + np.arange(x, dtype=int) for x in args.toys]
    )
    blinded_fits = [f == 0 or (f > 0 and args.toysDataMode == "observed") for f in fits]

    indata = inputdata.FitInputData(args.filename, args.pseudoData)

    poi_model = ph.load_model(args.physicsModel, indata, **vars(args))

    ifitter = fitter.Fitter(indata, poi_model, args, do_blinding=any(blinded_fits))

    # mappings for observables and parameters
    mappings = []
    for margs in args.mapping:
        mapping = mh.load_mapping(margs[0], indata, *margs[1:])
        mappings.append(mapping)

    if args.compositeMapping:
        mappings = [
            mp.CompositeMapping(mappings),
        ]

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # pass meta data into output file
    meta = {
        "meta_info": output_tools.make_meta_info_dict(args=args),
        "meta_info_input": ifitter.indata.metadata,
        "procs": ifitter.indata.procs,
        "pois": ifitter.poi_model.pois,
        "nois": ifitter.parms[ifitter.poi_model.npoi :][indata.noiidxs],
    }

    with workspace.Workspace(
        args.output,
        args.outname,
        postfix=args.postfix,
        fitter=ifitter,
    ) as ws:

        ws.write_meta(meta=meta)

        init_time = time.time()
        prefit_time = []
        postfit_time = []
        fit_time = []

        for i, ifit in enumerate(fits):
            group = ["results"]

            if args.pseudoData is None:
                datasets = zip([ifitter.indata.data_obs], [ifitter.indata.data_var])
            else:
                # shape nobs x npseudodata
                datasets = zip(
                    tf.transpose(indata.pseudodata_obs),
                    (
                        tf.transpose(indata.pseudodata_var)
                        if indata.pseudodata_var is not None
                        else [None] * indata.pseudodata_obs.shape[-1]
                    ),
                )

            # loop over (pseudo)data sets
            for j, (data_values, data_variances) in enumerate(datasets):

                ifitter.defaultassign()
                if ifit == -1:
                    group.append("asimov")
                    ifitter.set_nobs(ifitter.expected_yield())
                else:
                    if ifit == 0:
                        ifitter.set_nobs(data_values)
                    elif ifit >= 1:
                        group.append(f"toy{ifit}")
                        ifitter.toyassign(
                            data_values,
                            data_variances,
                            syst_randomize=args.toysSystRandomize,
                            data_randomize=args.toysDataRandomize,
                            data_mode=args.toysDataMode,
                            randomize_parameters=args.toysRandomizeParameters,
                        )

                if args.pseudoData is not None:
                    # label each pseudodata set
                    if j == 0:
                        group.append(indata.pseudodatanames[j])
                    else:
                        group[-1] = indata.pseudodatanames[j]

                ws.add_parms_hist(
                    values=ifitter.x,
                    variances=tf.linalg.diag_part(ifitter.cov),
                    hist_name="parms_prefit",
                )

                prefit_time.append(time.time())

                observations = data_values if ifit >= 0 else ifitter.expected_yield()

                do_asymptotic_limits(
                    args,
                    ifitter,
                    ws,
                    data_values=observations,
                    bkg_only_fit=ifit >= 0,
                    on_channel=args.limitsOnChannel,
                )
                fit_time.append(time.time())

                ws.dump_and_flush("_".join(group))
                postfit_time.append(time.time())

    end_time = time.time()
    logger.info(f"{end_time - start_time:.2f} seconds total time")
    logger.debug(f"{init_time - start_time:.2f} seconds initialization time")
    for i, ifit in enumerate(fits):
        logger.debug(f"For fit {ifit}:")
        dt = init_time if i == 0 else fit_time[i - 1]
        t0 = prefit_time[i] - dt
        t1 = fit_time[i] - prefit_time[i]
        t2 = postfit_time[i] - fit_time[i]
        logger.debug(f"{t0:.2f} seconds for prefit")
        logger.debug(f"{t1:.2f} seconds for fit")
        logger.debug(f"{t2:.2f} seconds for postfit")


if __name__ == "__main__":
    main()
