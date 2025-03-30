import argparse
import os

import hist
import numpy as np
import uproot
from tqdm import tqdm
from wums import logging

from combinetf2 import tensorwriter

logger = None


class DatacardParser:
    """
    Parser for Combine Datacard format files
    """

    def __init__(self):
        self.imax = None  # number of channels
        self.jmax = None  # number of backgrounds
        self.kmax = None  # number of nuisance parameters
        self.bins = []  # list of bin/channel names
        self.observations = {}  # observations for each bin
        self.processes = []  # list of process names
        self.process_indices = {}  # numerical indices for processes
        self.rates = {}  # expected rates for each bin and process
        self.systematics = []  # list of systematic uncertainties
        self.shapes = []  # shape directives
        self.bin_process_map = {}  # mapping of bins to processes
        self.param_lines = []  # param directives
        self.rate_params = []  # rate param directives
        self.group_lines = []  # group directives
        self.nuisance_edits = []  # nuisance edit directives
        self.max_depth = (
            0  # how many directories are there to find the histograms in the root files
        )

    def parse_file(self, filename):
        """Parse a datacard file"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Datacard file not found: {filename}")

        with open(filename, "r") as file:
            content = file.read()

        # Remove comments (lines starting with #)
        lines = [line.strip() for line in content.split("\n")]
        lines = [line for line in lines if line and not line.startswith("#")]

        # Parse header section (imax, jmax, kmax)
        self._parse_header(lines)

        # Parse bin and observation section
        self._parse_observations(lines)

        # Parse process and rate section
        self._parse_processes_and_rates(lines)

        # Parse shapes section
        self._parse_shapes(lines, filename)

        # Parse systematics section
        self._parse_systematics(lines)

        # Parse additional directives
        self._parse_additional_directives(lines)

        return self

    def _parse_header(self, lines):
        """Parse the header section with imax, jmax, kmax"""
        for line in lines:
            if line.startswith("imax"):
                parts = line.split()
                self.imax = parts[1] if parts[1] != "*" else None
            elif line.startswith("jmax"):
                parts = line.split()
                self.jmax = parts[1] if parts[1] != "*" else None
            elif line.startswith("kmax"):
                parts = line.split()
                self.kmax = parts[1] if parts[1] != "*" else None

    def _parse_observations(self, lines):
        """Parse the bin and observation section"""
        bin_line = None
        obs_line = None

        for i, line in enumerate(lines):
            if (
                line.startswith("bin")
                and not bin_line
                and not self._is_process_bin_line(lines, i)
            ):
                bin_line = line
            elif line.startswith("observation") and not obs_line:
                obs_line = line

        if bin_line and obs_line:
            bin_parts = bin_line.split()[1:]
            obs_parts = obs_line.split()[1:]

            self.bins = bin_parts

            if len(bin_parts) == len(obs_parts):
                for i, bin_name in enumerate(bin_parts):
                    self.observations[bin_name] = float(obs_parts[i])

    def _is_process_bin_line(self, lines, index):
        """Check if a bin line is part of the process section"""
        if index > 0 and index < len(lines) - 2:
            next_line = lines[index + 1]
            if next_line.startswith("process"):
                return True
        return False

    def _parse_processes_and_rates(self, lines):
        """Parse the process and rate section"""
        bin_line = None
        process_name_line = None
        process_index_line = None
        rate_line = None

        # Find the three consecutive lines
        for i in range(len(lines) - 3):
            if (
                lines[i].startswith("bin")
                and lines[i + 1].startswith("process")
                and lines[i + 2].startswith("process")
                and lines[i + 3].startswith("rate")
            ):
                bin_line = lines[i]
                process_name_line = lines[i + 1]
                process_index_line = lines[i + 2]
                rate_line = lines[i + 3]
                break

        if bin_line and process_name_line and process_index_line and rate_line:
            bins = bin_line.split()[1:]
            process_names = process_name_line.split()[1:]
            process_indices = process_index_line.split()[1:]
            rates = rate_line.split()[1:]

            self.processes = list(
                dict.fromkeys(process_names)
            )  # Remove duplicates while preserving order

            # Map process names to their indices
            for name, idx in zip(process_names, process_indices):
                self.process_indices[name] = int(idx)

            # Map bins to processes with rates
            for i in range(len(bins)):
                bin_name = bins[i]
                process_name = process_names[i]
                rate = float(rates[i])

                if bin_name not in self.bin_process_map:
                    self.bin_process_map[bin_name] = {}

                self.bin_process_map[bin_name][process_name] = rate

                key = (bin_name, process_name)
                self.rates[key] = rate

    def _parse_shapes(self, lines, filename):
        """Parse shape directives"""
        for line in lines:
            if line.startswith("shapes"):
                parts = line.split()
                if len(parts) >= 5:
                    file_path = parts[3]

                    # save absolute path of shapes
                    if not os.path.isabs(file_path):
                        datacard_dir = os.path.dirname(os.path.abspath(filename))
                        file_path = os.path.join(datacard_dir, file_path)

                    if not os.path.exists(file_path):
                        raise FileNotFoundError(
                            f"ROOT file with shapes not found: {file_path}"
                        )

                    shape_info = {
                        "process": parts[1],
                        "channel": parts[2],
                        "file": file_path,
                        "histogram_pattern": parts[4],
                    }
                    self.max_depth = max(self.max_depth, parts[4].count("/"))
                    if len(parts) >= 6:
                        shape_info["histogram_syst_pattern"] = parts[5]
                    self.shapes.append(shape_info)

    def _parse_systematics(self, lines):
        """Parse systematic uncertainty specifications"""
        # Skip lines until after the rate line
        rate_index = None
        for i, line in enumerate(lines):
            if line.startswith("rate"):
                rate_index = i
                break

        if rate_index is None:
            return

        # Parse systematics after the rate line
        for i in range(rate_index + 1, len(lines)):
            line = lines[i]
            # Stop when we hit other directives
            if any(
                line.startswith(directive)
                for directive in [
                    "shapes",
                    "nuisance",
                    "param",
                    "rateParam",
                    "group",
                    "extArg",
                ]
            ):
                break

            parts = line.split()
            if len(parts) < 2:
                continue

            # Check if this looks like a systematic entry (name followed by type)
            if parts[1] in ["lnN", "shape", "gmN", "lnU", "shapeN", "shape?", "shapeU"]:
                syst_name = parts[0]
                syst_type = parts[1]

                syst_info = {"name": syst_name, "type": syst_type, "effects": {}}

                # If gmN, the next value is N
                if syst_type == "gmN" and len(parts) > 2:
                    syst_info["n_events"] = int(parts[2])
                    effects_start = 3
                else:
                    effects_start = 2

                # Get effects on each process
                if len(parts) > effects_start:
                    effects = parts[effects_start:]
                    bin_process_pairs = list(self.rates.keys())

                    if len(effects) == len(bin_process_pairs):
                        for j, effect in enumerate(effects):
                            bin_name, process_name = bin_process_pairs[j]
                            if effect != "-":
                                syst_info["effects"][(bin_name, process_name)] = effect

                self.systematics.append(syst_info)

    def _parse_additional_directives(self, lines):
        """Parse additional directives like param, rateParam, group, etc."""
        for line in lines:
            if line.startswith("param"):
                self.param_lines.append(line)
            elif line.startswith("rateParam"):
                self.rate_params.append(line)
            elif line.startswith("group"):
                self.group_lines.append(line)
            elif line.startswith("nuisance edit"):
                self.nuisance_edits.append(line)

    def get_summary(self):
        """Return a summary of the parsed datacard"""
        summary = {
            "channels": self.imax if self.imax else len(self.bins),
            "backgrounds": (
                self.jmax
                if self.jmax
                else len(
                    [p for p in self.processes if self.process_indices.get(p, 0) > 0]
                )
            ),
            "systematics": self.kmax if self.kmax else len(self.systematics),
            "bins": self.bins,
            "observations": self.observations,
            "processes": self.processes,
            "signal_processes": [
                p for p in self.processes if self.process_indices.get(p, 0) <= 0
            ],
            "background_processes": [
                p for p in self.processes if self.process_indices.get(p, 0) > 0
            ],
            "systematics_count": len(self.systematics),
            "has_shapes": len(self.shapes) > 0,
        }
        return summary


class DatacardConverter:
    """
    Convert data from Combine datacards and ROOT files to hdf5 tensor format
    """

    def __init__(self, datacard_file, mass="125.38"):
        """
        Initialize the converter with a datacard file

        Args:
            datacard_file: Path to the datacard file
        """
        self.datacard_file = datacard_file
        self.parser = DatacardParser()
        self.root_directories = {}  # Cache for opened ROOT files and their directories
        self.mass = mass

        # For counting experiments
        self.yield_axis = hist.axis.Integer(
            0, 1, name="yield", overflow=False, underflow=False
        )

    def parse(self):
        """Parse the datacard file"""
        self.parser.parse_file(self.datacard_file)
        return self

    def load_root_directories(self, file_path):
        """Get a ROOT file, opening it if not already open"""
        if file_path not in self.root_directories:
            # Check if path is relative to the datacard location
            datacard_dir = os.path.dirname(os.path.abspath(self.datacard_file))
            if not os.path.isabs(file_path):
                file_path = os.path.join(datacard_dir, file_path)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"ROOT file not found: {file_path}")

            logger.info(f"Open file {file_path}")
            file = uproot.open(file_path)

            search_depth = self.parser.max_depth - 1
            if search_depth >= 0:
                # recursively find all directories up to a depth of 'search_depth', setting 'search_depth' to a proper value speeds up things (if you know how deep the directories go)
                def get_directories(directory, level=0):
                    keys = set([k.split("/")[0] for k in directory.keys()])
                    directories = [
                        d
                        for d in keys
                        if isinstance(directory[d], uproot.reading.ReadOnlyDirectory)
                    ]
                    if len(directories) == 0:
                        return directory
                    elif level >= search_depth:
                        return {d: directory[d] for d in directories}
                    else:
                        return {
                            d: get_directories(directory[d], level=level + 1)
                            for d in directories
                        }

                self.root_directories[file_path] = get_directories(file)
            else:
                self.root_directories[file_path] = file

    def get_histogram(
        self, shape_info, process, bin_name, systematic=None, demand=True
    ):
        """
        Get a histogram based on shape information

        Args:
            shape_info: Shape directive information
            process: Process name
            bin_name: Bin/channel name
            systematic: Systematic uncertainty (None for nominal)
            demand: throw an error if shape is not found and demand=True, otherwise not

        Returns:
            TH1 histogram or None if not found
        """
        if not shape_info:
            return None

        file_path = shape_info["file"]

        if systematic and "histogram_syst_pattern" in shape_info:
            syst_pattern = shape_info["histogram_syst_pattern"]
            hist_name = syst_pattern.replace("$PROCESS", process)
            hist_name = hist_name.replace("$CHANNEL", bin_name)
            hist_name = hist_name.replace("$SYSTEMATIC", systematic)
            hist_name = hist_name.replace("$MASS", self.mass)
        else:
            # Replace variables in histogram pattern
            hist_pattern = shape_info["histogram_pattern"]
            hist_name = hist_pattern.replace("$PROCESS", process)
            hist_name = hist_name.replace("$CHANNEL", bin_name)
            hist_name = hist_name.replace("$MASS", self.mass)

            if systematic:
                # If no separate systematic pattern, assume appending Up/Down to nominal
                hist_name = f"{hist_name}_{systematic}"

        # Try to find the histogram in the file
        # histogram = self.get_root_directories(file_path)
        histogram = self.root_directories[file_path]
        for p in hist_name.split("/"):
            histogram = histogram.get(p, {})

        if not histogram or not histogram.classname.startswith("TH1"):
            if demand:
                raise ValueError(f"Histogram {hist_name} not found in {file_path}")
            else:
                logger.debug(
                    f"Histogram {hist_name} not found in {file_path}, but 'demand=False'. Skip it"
                )
                return None
        else:
            return histogram.to_boost()

    def convert_to_hdf5(self, sparse=False):
        """
        Convert the datacard and histograms to numpy arrays

        Returns:
            Dictionary of numpy arrays with all the data
        """
        logger.info("Parse datacard text")
        self.parse()
        logger.info("Prepare histograms")
        # Get shape directives for each process and bin
        shape_map = {}
        for shape in self.parser.shapes:
            process = shape["process"]
            channel = shape["channel"]

            # Handle wildcards
            if process == "*":
                processes = ["data_obs", *self.parser.processes]
            else:
                processes = [process]

            if channel == "*":
                channels = self.parser.bins
            else:
                channels = [channel]

            for p in processes:
                for c in channels:
                    shape_map[(p, c)] = shape

            self.load_root_directories(shape["file"])

        logger.info("Convert histograms into hdf5 tensor")

        # TODO: rate params
        # TODO: nuisance groups

        writer = tensorwriter.TensorWriter(
            sparse=sparse,
        )

        logger.info("loop over channels (aka combine bins) for nominal histograms")
        for bin_name in tqdm(self.parser.bins, desc="Processing"):

            for process_name in ["data_obs", *self.parser.bin_process_map[bin_name]]:

                shape_info = (
                    shape_map.get((process_name, bin_name))
                    or shape_map.get(("*", bin_name))
                    or shape_map.get((process_name, "*"))
                    or shape_map.get(("*", "*"))
                )

                if shape_info:
                    h_proc = self.get_histogram(shape_info, process_name, bin_name)
                else:
                    # For counting experiments
                    h_proc = hist.Hist(
                        self.yield_axis,
                        data=np.array([self.parser.observations.get(bin_name, 0)]),
                        storage=hist.storage.Double(),
                    )

                if process_name == "data_obs":
                    writer.add_channel(h_proc.axes, bin_name)
                    writer.add_data(h_proc, bin_name)
                else:
                    writer.add_process(
                        h_proc,
                        process_name,
                        bin_name,
                        signal=self.parser.process_indices.get(process_name, 0) <= 0,
                    )

        def add_lnN_syst(writer, name, process, channel, effect):
            # Parse the effect (could be asymmetric like 0.9/1.1)
            if "/" in effect:
                down, up = effect.split("/")
                writer.add_lnN_systematic(
                    name, process, channel, [(float(up), float(down))]
                )
            else:
                writer.add_lnN_systematic(name, process, channel, float(effect))

        logger.info("loop over systematic variations")
        for syst in tqdm(self.parser.systematics, desc="Processing"):

            if syst["type"] in ["shape", "shapeN", "shape?", "shapeU"]:
                # TODO dedicated treatment for shapeN
                for (bin_name, process_name), effect in syst["effects"].items():
                    if effect not in ["-", "0"]:

                        shape_info = (
                            shape_map.get((process_name, bin_name))
                            or shape_map.get(("*", bin_name))
                            or shape_map.get((process_name, "*"))
                            or shape_map.get(("*", "*"))
                        )

                        hist_up = self.get_histogram(
                            shape_info,
                            process_name,
                            bin_name,
                            f"{syst['name']}Up",
                            demand=syst["type"] != "shape?",
                        )
                        hist_down = self.get_histogram(
                            shape_info,
                            process_name,
                            bin_name,
                            f"{syst['name']}Down",
                            demand=syst["type"] != "shape?",
                        )

                        if hist_up is None and hist_down is None:
                            # 'syst?' case
                            add_lnN_syst(
                                writer, syst["name"], process_name, bin_name, effect
                            )
                        else:
                            writer.add_systematic(
                                [hist_up, hist_down],
                                syst["name"],
                                process_name,
                                bin_name,
                                kfactor=float(effect),
                                constrained=syst["type"]
                                != "shapeU",  # TODO check if shapeU is unconstriained
                            )

            elif syst["type"] in ["lnN", "lnU"]:
                # TODO dedicated treatment for lnU

                for (bin_name, process_name), effect in syst["effects"].items():
                    if effect not in ["-", "0"]:
                        add_lnN_syst(
                            writer, syst["name"], process_name, bin_name, effect
                        )

        return writer


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
        "--mass",
        type=str,
        default="125.38",
        help="Higgs boson mass to replace $MASS string in datacard",
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

    converter = DatacardConverter(args.datacard, args.mass)

    writer = converter.convert_to_hdf5(sparse=args.sparse)

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


if __name__ == "__main__":
    main()
