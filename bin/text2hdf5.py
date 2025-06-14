#!/usr/bin/env python3

import argparse
import os
from pprint import pprint

from wums import logging

from rabbit.datacard_converter import DatacardConverter

logger = None


def main():
    parser = argparse.ArgumentParser(
        description="Convert Combine datacard and ROOT files to different formats"
    )
    parser.add_argument("datacard", help="Path to the datacard file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="output directory, if 'None' same as input datacard",
    )
    parser.add_argument(
        "--outname",
        default=None,
        help="output file name, if 'None' same as input datacard but with .hdf5 extension",
    )
    parser.add_argument(
        "--postfix",
        default=None,
        type=str,
        help="Postfix to append on output file name",
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
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4],
        help="Set verbosity level with logging, the larger the more verbose",
    )
    parser.add_argument(
        "--noColorLogger", action="store_true", help="Do not use logging with colors"
    )

    args = parser.parse_args()

    global logger
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    converter = DatacardConverter(
        args.datacard, use_root=args.root, mass=args.mass, symmetrize=args.symmetrize
    )
    writer = converter.convert_to_hdf5(sparse=args.sparse)

    pprint(converter.parser.get_summary())

    directory = args.output
    if directory is None:
        directory = os.path.dirname(args.datacard)
    if directory == "":
        directory = "./"
    filename = args.outname
    if filename is None:
        filename = os.path.splitext(os.path.basename(args.datacard))[0]
    if args.postfix:
        filename += f"_{args.postfix}"
    writer.write(outfolder=directory, outfilename=filename)

    del converter


if __name__ == "__main__":
    main()
