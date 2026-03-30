#!/usr/bin/env python3

import argparse
import os
from pprint import pprint

from wums import logging

from rabbit import parsing
from rabbit.datacard_converter import DatacardConverter

logger = None


def make_parser():
    parser = argparse.ArgumentParser(
        description="Convert Combine datacard and ROOT files to different formats"
    )
    parsing._add_base_args(parser)
    parsing._add_output_args(parser)
    parser.add_argument("datacard", help="Path to the datacard file")
    parser.add_argument(
        "--outname",
        default=None,
        help="output file name, if 'None' same as input datacard but with .hdf5 extension",
    )
    parser.add_argument(
        "--sparse",
        default=False,
        action="store_true",
        help="Make sparse tensor",
    )
    parser.add_argument(
        "--symmetrize",
        default=None,
        choices=[None, "conservative", "average", "linear", "quadratic"],
        type=str,
        help="Symmetrize tensor by forcing systematics to 'average'",
    )
    parser.add_argument(
        "--mass",
        type=str,
        default="125.38",
        help="Higgs boson mass to replace $MASS string in datacard",
    )
    parser.add_argument(
        "--root",
        action="store_true",
        help="Use root to load histograms, otherwise uproot",
    )
    return parser


def main():
    args = make_parser().parse_args()

    global logger
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    converter = DatacardConverter(
        args.datacard, use_root=args.root, mass=args.mass, symmetrize=args.symmetrize
    )
    writer = converter.convert_to_hdf5(sparse=args.sparse)

    pprint(converter.parser.get_summary())

    directory = args.outpath
    filename = args.outname
    if filename is None:
        filename = os.path.splitext(os.path.basename(args.datacard))[0]
    if args.postfix:
        filename += f"_{args.postfix}"
    writer.write(outfolder=directory, outfilename=filename)

    del converter


if __name__ == "__main__":
    main()
