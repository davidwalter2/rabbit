import os

from wums import logging

logger = logging.child_logger(__name__)


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
            logger.debug(f"Now at line {i}: {line}")
            parts = line.split()

            if len(parts) < 2:
                continue

            # Stop when we hit other directives
            if parts[1] in [
                "shapes",
                "nuisance",
                "param",
                "rateParam",
                "group",
                "extArg",
            ]:
                break

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
            parts = line.split()
            if len(parts) < 2:
                continue
            reduced_parts = [parts[0], *parts[2:]]
            if parts[1] == "param":
                self.param_lines.append(reduced_parts)
            elif parts[1] == "rateParam":
                self.rate_params.append(reduced_parts)
            elif parts[1] == "group":
                self.group_lines.append(reduced_parts)
            elif parts[1] == "nuisance edit":
                self.nuisance_edits.append(reduced_parts)

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
            "rate_params": self.rate_params,
            "param_lines": self.param_lines,
            "group_lines": self.group_lines,
            "nuisance_edits": self.nuisance_edits,
        }
        return summary
