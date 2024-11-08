#!/usr/bin/env python3

from datetime import datetime, timedelta
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import polars as pl


# embed fonts into the final pdf
mpl.rcParams["pdf.fonttype"] = 42


##################################################
# general configuration
##################################################
# HPL log file as produced by
# - "HPLinpack 2.3  --  High-Performance Linpack benchmark  --   December 2, 2018"
# - "HPLinpack 2.1  --  High-Performance Linpack benchmark  --   October 26, 2012"
hpl_log_path = Path(f"hpl.log")
# rpeak per node: (96 cores) * (2.8 GHz experimental AVX512 boost frequency for 1 full CPU à 48 cores) * (2 FMA units) * (1 multiply + 1 add) * (512 vector width / 64-bit floating point number)
rpeak_gflops_per_node = 96 * 2.800 * 2 * 2 * (512 / 64)
# specify plotting configuration which ensures sufficient space at the top left for summary box
# --> this value may get automatically adjusted to align the final axis tick with the axis limit
power_kilowatt_limit = 1100

# time resolution needed for detecting missing values
pdus_resolution = "5s"
powermeter_resolution = "1s"
# the common resolution is used to resample the data for visualization (should be at least the maximum of the measurement resolutions)
common_resolution = "5s"
# configure idle phase reference time frame
idle_start = int(datetime(2024, 4, 24, 3, 0, 0, tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
idle_end = int(idle_start + timedelta(hours=2).total_seconds())
# control output path
pdus_path = Path("pdus.parquet")
powermeter_path = Path("powermeter.parquet")
output_directory_path = Path(".")
output_directory_path.mkdir(exist_ok=True)
output_path = Path(output_directory_path / "summary.txt")
output_file = output_path.open(mode="w")


# SLURM job data with start and end time
jobid = 45056267
job_start, job_end = 1713899471, 1713915850
num_nodes = 598
racks = {
    # compute racks
    143,
    144,
    146,
    147,
    148,
    149,
    243,
    244,
    245,
    246,
    247,
    248,
    249,
    # global cooling equipment and network (side coolers, CDUs, switches)
    100,
    200,
    # management rack for SLURM controller and login nodes
    145,
    # isilon storage
    443,
    444,
}
num_racks = len(racks)

print(
    f"using data from {num_nodes} nodes in the following {num_racks} racks: {sorted(racks)}",
    file=output_file,
)

##################################################
# HPL output parsing
##################################################
hpl_log_text = hpl_log_path.read_text()

num_tests_pattern = re.compile(r"(\d+) tests with the following results")
num_tests_match = num_tests_pattern.search(hpl_log_text)
if not num_tests_match:
    print(f"WARNING: could not find the number of tests in '{hpl_log_path}'...", file=sys.stderr)
else:
    num_tests = int(num_tests_match.group(1))
    if num_tests != 1:
        print(f"WARNING: found {num_tests} tests instead of exactly 1 test in '{hpl_log_path}'...", file=sys.stderr)

core_start_pattern = re.compile(r"HPL_pdgesv\(\) start time\s+(.*)")
core_end_pattern = re.compile(r"HPL_pdgesv\(\) end time\s+(.*)")

core_start_match = core_start_pattern.search(hpl_log_text)
core_end_match = core_end_pattern.search(hpl_log_text)
if not core_start_match or not core_end_match:
    print(f"ERROR: could not find core phase start and end time in '{hpl_log_path}'...", file=sys.stderr)
    sys.exit(1)

# fmt: off
core_start = int(datetime.strptime(core_start_match.group(1), r"%c").replace(tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
core_end = int(datetime.strptime(core_end_match.group(1), r"%c").replace(tzinfo=ZoneInfo("Europe/Berlin")).timestamp())
# fmt: on

hpl_log_lines = hpl_log_text.splitlines()
hpl_log_lines = [line for line in hpl_log_lines if line]
for index, line in enumerate(hpl_log_lines):
    if "HPL_pdgesv() start time" in line:
        break
else:
    print(f"ERROR: could not find GFLOPS in '{hpl_log_path}'...", file=sys.stderr)
rmax_gflops = float(hpl_log_lines[index - 1].split()[6])
rpeak_gflops = num_nodes * rpeak_gflops_per_node
percent_rmax_of_rpeak = 100 * rmax_gflops / rpeak_gflops

##################################################
# energy data parsing
##################################################
pdus = pl.read_parquet(pdus_path).with_columns(
    pl.col("time").cast(pl.String).str.to_datetime(time_zone="Europe/Berlin"),
    pl.col("rack").cast(pl.Int64),
    pl.col("num").cast(pl.Int64),
    pl.col("energy").cast(pl.Float64),
)

powermeter = pl.read_parquet(powermeter_path).with_columns(
    pl.col("time").cast(pl.String).str.to_datetime(time_zone="Europe/Berlin"),
    pl.col("channel").cast(pl.Int64),
    pl.col("energy").cast(pl.Float64),
)

pdus = (
    pdus.with_columns(
        pdu=pl.concat_str(pl.col("rack"), pl.col("num"), separator="#"),
    )
    .pivot(
        on="pdu",
        values="energy",
        index="time",
    )
    .sort(by="time")
    .upsample("time", every=pdus_resolution)
)

powermeter = powermeter.pivot(
    on="channel",
    values="energy",
    index="time",
).sort(by="time")

# write reference files of raw data for Green500 submission
pdus_job = pdus.filter(
    pl.col("time").dt.timestamp("ns").is_between(job_start * 10**9, job_end * 10**9, closed="both")
).sort(by="time")
pdus_job.write_csv(output_directory_path / "pdus-job.csv")

pdus_core = pdus.filter(
    pl.col("time").dt.timestamp("ns").is_between(core_start * 10**9, core_end * 10**9, closed="both")
).sort(by="time")
pdus_core.write_csv(output_directory_path / "pdus-core.csv")

pdus_idle = pdus.filter(
    pl.col("time").dt.timestamp("ns").is_between(idle_start * 10**9, idle_end * 10**9, closed="both")
).sort(by="time")
pdus_idle.write_csv(output_directory_path / "pdus-idle.csv")

powermeter_job = powermeter.filter(
    pl.col("time").dt.timestamp("ns").is_between(job_start * 10**9, job_end * 10**9, closed="both")
).sort(by="time")
powermeter_job.write_csv(output_directory_path / "powermeter-job.csv")

powermeter_core = powermeter.filter(
    pl.col("time").dt.timestamp("ns").is_between(core_start * 10**9, core_end * 10**9, closed="both")
).sort(by="time")
powermeter_core.write_csv(output_directory_path / "powermeter-core.csv")

powermeter_idle = powermeter.filter(
    pl.col("time").dt.timestamp("ns").is_between(idle_start * 10**9, idle_end * 10**9, closed="both")
).sort(by="time")
powermeter_idle.write_csv(output_directory_path / "powermeter-idle.csv")

##################################################
# check missing data and repair it
##################################################
pdus_missing = (
    pdus.null_count()
    .transpose(include_header=True, header_name="PDU", column_names=["Null Count"])
    .filter(pl.col("Null Count") > 0)
    .sort("PDU")
)

num_missing = pdus_missing.sum()["Null Count"][0]
num_expected = len(pdus) * (len(pdus.columns) - 1)
percent_missing = 100 * num_missing / num_expected
print(
    f"(pdus) number of missing data points before reparation: {num_missing} / {num_expected} = {percent_missing:.3f}%",
    file=output_file,
)

pdus = pdus.with_columns(pl.exclude("time").interpolate_by("time"))

pdus_missing = (
    pdus.null_count()
    .transpose(include_header=True, header_name="PDU", column_names=["Null Count"])
    .filter(pl.col("Null Count") > 0)
    .sort("PDU")
)
num_missing = pdus_missing.sum()["Null Count"][0]
num_expected = len(pdus) * (len(pdus.columns) - 1)
percent_missing = 100 * num_missing / num_expected
print(
    f"(pdus) number of missing data points after reparation: {num_missing} / {num_expected} = {percent_missing:.3f}%",
    file=output_file,
)

assert pdus_missing.is_empty()

# since the powermeter data is not recorded at exactly the same time intervals just check that all or nothing is missing each timestamp
assert (
    powermeter.select(
        any_missing=pl.any_horizontal(pl.exclude("time").is_null()),
        all_missing=pl.all_horizontal(pl.exclude("time").is_null()),
    )
    .filter(pl.col("any_missing") & ~pl.col("all_missing"))
    .is_empty()
)

##################################################
# post-processing and aggregations
##################################################
pdus_sum = pdus.select(
    time=pl.col("time"),
    energy=pl.sum_horizontal(pl.exclude("time")),
)

powermeter_sum = powermeter.select(
    time=pl.col("time"),
    energy=pl.sum_horizontal(pl.exclude("time")),
)

pdus_sum_job = pdus_sum.filter(
    pl.col("time").dt.timestamp("ns").is_between(job_start * 10**9, job_end * 10**9, closed="both")
).sort(by="time")

pdus_sum_core = pdus_sum.filter(
    pl.col("time").dt.timestamp("ns").is_between(core_start * 10**9, core_end * 10**9, closed="both")
).sort(by="time")

pdus_sum_idle = pdus_sum.filter(
    pl.col("time").dt.timestamp("ns").is_between(idle_start * 10**9, idle_end * 10**9, closed="both")
).sort(by="time")

powermeter_sum_job = powermeter_sum.filter(
    pl.col("time").dt.timestamp("ns").is_between(job_start * 10**9, job_end * 10**9, closed="both")
).sort(by="time")

powermeter_sum_core = powermeter_sum.filter(
    pl.col("time").dt.timestamp("ns").is_between(core_start * 10**9, core_end * 10**9, closed="both")
).sort(by="time")

powermeter_sum_idle = powermeter_sum.filter(
    pl.col("time").dt.timestamp("ns").is_between(idle_start * 10**9, idle_end * 10**9, closed="both")
).sort(by="time")

##################################################
# calculate final metrics
##################################################
runtime_seconds_job = job_end - job_start
runtime_seconds_core = core_end - core_start
runtime_seconds_idle = idle_end - idle_start

# instead of the actual start and end times, the Green500 methodology uses the first and last collected energy value inside the time frame
runtime_seconds_job_pdus = (pdus_sum_job["time"][-1] - pdus_sum_job["time"][0]).total_seconds()
energy_watthours_job_pdus = pdus_sum_job["energy"][-1] - pdus_sum_job["energy"][0]
power_watt_job_pdus = 60 * 60 * energy_watthours_job_pdus / runtime_seconds_job_pdus
runtime_seconds_job_powermeter = (powermeter_sum_job["time"][-1] - powermeter_sum_job["time"][0]).total_seconds()
energy_watthours_job_powermeter = powermeter_sum_job["energy"][-1] - powermeter_sum_job["energy"][0]
power_watt_job_powermeter = 60 * 60 * energy_watthours_job_powermeter / runtime_seconds_job_powermeter
power_watt_job = power_watt_job_pdus + power_watt_job_powermeter

runtime_seconds_core_pdus = (pdus_sum_core["time"][-1] - pdus_sum_core["time"][0]).total_seconds()
energy_watthours_core_pdus = pdus_sum_core["energy"][-1] - pdus_sum_core["energy"][0]
power_watt_core_pdus = 60 * 60 * energy_watthours_core_pdus / runtime_seconds_core_pdus
runtime_seconds_core_powermeter = (powermeter_sum_core["time"][-1] - powermeter_sum_core["time"][0]).total_seconds()
energy_watthours_core_powermeter = powermeter_sum_core["energy"][-1] - powermeter_sum_core["energy"][0]
power_watt_core_powermeter = 60 * 60 * energy_watthours_core_powermeter / runtime_seconds_core_powermeter
power_watt_core = power_watt_core_pdus + power_watt_core_powermeter

runtime_seconds_idle_pdus = (pdus_sum_idle["time"][-1] - pdus_sum_idle["time"][0]).total_seconds()
energy_watthours_idle_pdus = pdus_sum_idle["energy"][-1] - pdus_sum_idle["energy"][0]
power_watt_idle_pdus = 60 * 60 * energy_watthours_idle_pdus / runtime_seconds_idle_pdus
runtime_seconds_idle_powermeter = (powermeter_sum_idle["time"][-1] - powermeter_sum_idle["time"][0]).total_seconds()
energy_watthours_idle_powermeter = powermeter_sum_idle["energy"][-1] - powermeter_sum_idle["energy"][0]
power_watt_idle_powermeter = 60 * 60 * energy_watthours_idle_powermeter / runtime_seconds_idle_powermeter
power_watt_idle = power_watt_idle_pdus + power_watt_idle_powermeter

energy_efficiency_core = rmax_gflops / power_watt_core

# fmt: off
print(f"Job with SLURM ID {jobid}:", file=output_file)
print(f"- Start: {job_start} = {datetime.fromtimestamp(job_start, tz=ZoneInfo('Europe/Berlin'))}", file=output_file)
print(f"- End: {job_end} = {datetime.fromtimestamp(job_end, tz=ZoneInfo('Europe/Berlin'))}", file=output_file)
print(f"- Runtime: {int(runtime_seconds_job)} s ≈ {runtime_seconds_job / 60 / 60:.3f} h", file=output_file)
print(f"  - Runtime pdus: {runtime_seconds_job_pdus:.3f} s", file=output_file)
print(f"  - Runtime powermeter: {runtime_seconds_job_powermeter:.3f} s", file=output_file)
print(f"- Average Power: {power_watt_job:.3f} W", file=output_file)
print(f"Core:", file=output_file)
print(f"- Start: {core_start} = {datetime.fromtimestamp(core_start, tz=ZoneInfo('Europe/Berlin'))}", file=output_file)
print(f"- End: {core_end} = {datetime.fromtimestamp(core_end, tz=ZoneInfo('Europe/Berlin'))}", file=output_file)
print(f"- Runtime: {int(runtime_seconds_core)} s ≈ {runtime_seconds_core / 60 / 60:.3f} h", file=output_file)
print(f"  - Runtime pdus: {runtime_seconds_core_pdus:.3f} s", file=output_file)
print(f"  - Runtime powermeter: {runtime_seconds_core_powermeter:.3f} s", file=output_file)
print(f"- Average Power: {power_watt_core:.3f} W", file=output_file)
print(f"Top500: Rmax / Rpeak = {rmax_gflops / 1_000:.3f} TFLOPS / {rpeak_gflops / 1_000:.3f} TFLOPS = {percent_rmax_of_rpeak:.3f}%", file=output_file)
print(f"Green500: Rmax / Core Average Power = {rmax_gflops / 1_000:.3f} TFLOPS / {power_watt_core / 1_000:.3f} kW = {energy_efficiency_core:.3f} GFLOPS/Watt", file=output_file)
print(f"Idle:", file=output_file)
print(f"- Start: {idle_start} = {datetime.fromtimestamp(idle_start, tz=ZoneInfo('Europe/Berlin'))}", file=output_file)
print(f"- End: {idle_end} = {datetime.fromtimestamp(idle_end, tz=ZoneInfo('Europe/Berlin'))}", file=output_file)
print(f"- Runtime: {int(runtime_seconds_idle)} s ≈ {runtime_seconds_idle / 60 / 60:.3f} h", file=output_file)
print(f"  - Runtime pdus: {runtime_seconds_idle_pdus:.3f} s", file=output_file)
print(f"  - Runtime powermeter: {runtime_seconds_idle_powermeter:.3f} s", file=output_file)
print(f"- Average Power: {power_watt_idle:.3f} W", file=output_file)
# fmt: on

##################################################
# create final summary visualization
##################################################
powermeter_aligned = (
    powermeter_sum.with_columns(
        interpolated=pl.lit(False),
    )
    .vstack(
        powermeter_sum.select(
            time=pl.datetime_range(
                pl.col("time").min().dt.truncate(common_resolution),
                pl.col("time").max().dt.truncate(common_resolution),
                interval=common_resolution,
            ),
            energy=pl.lit(None),
            interpolated=pl.lit(True),
        )
    )
    .sort("time")
    .with_columns(pl.col("energy").interpolate_by("time"))
    .drop_nulls()
    .filter(pl.col("interpolated"))
    .drop("interpolated")
    .filter(pl.col("time").dt.epoch(time_unit="ns").is_between(job_start * 10**9, job_end * 10**9, closed="both"))
    .sort(by="time")
)

pdus_aligned = (
    pdus_sum.with_columns(
        interpolated=pl.lit(False),
    )
    .vstack(
        pdus_sum.select(
            time=pl.datetime_range(
                pl.col("time").min().dt.truncate(common_resolution),
                pl.col("time").max().dt.truncate(common_resolution),
                interval=common_resolution,
            ),
            energy=pl.lit(None),
            interpolated=pl.lit(True),
        )
    )
    .sort("time")
    .with_columns(pl.col("energy").interpolate_by("time"))
    .drop_nulls()
    .filter(pl.col("interpolated"))
    .drop("interpolated")
    .filter(pl.col("time").dt.epoch(time_unit="ns").is_between(job_start * 10**9, job_end * 10**9, closed="both"))
    .sort(by="time")
)

all_aligned = pdus_aligned.vstack(powermeter_aligned).group_by("time").agg(pl.col("energy").sum()).sort("time")

all_aligned = all_aligned.with_columns(
    energy_diff=pl.col("energy").diff(),
    time_diff=pl.col("time").diff(),
).select(
    time=pl.col("time"),
    # normalize energy in Wh to energy in kWh
    energy=(pl.col("energy") - pl.col("energy").first()) / 1_000,
    # normalize energy in Wh to power in kW
    power=pl.col("energy_diff") / 1_000 / (pl.col("time_diff").dt.total_nanoseconds() / 10**9 / 60 / 60),
)

axis_left: Axes
figure, axis_left = plt.subplots(nrows=1, ncols=1)
plot_left = axis_left.plot(all_aligned["time"], all_aligned["power"])[0]
axis_right = axis_left.twinx()
plot_right = axis_right.plot(all_aligned["time"], all_aligned["energy"])[0]

padding = 0.025
axis_left.text(
    padding,
    1 - padding,
    f"""Rpeak: {rpeak_gflops / 1_000:.3f} TFLOPS
Rmax: {rmax_gflops / 1_000:.3f} TFLOPS ({percent_rmax_of_rpeak:.3f}%)
Energy Efficiency: {energy_efficiency_core:.3f} GFLOPS/W
Core Average Power: {power_watt_core / 1_000:.3f} kW
Job Average Power: {power_watt_job / 1_000:.3f} kW
Idle Average Power: {power_watt_idle / 1_000:.3f} kW""",
    ha="left",
    va="top",
    ma="left",
    transform=axis_left.transAxes,
    bbox=dict(facecolor="white", edgecolor="#dedede", boxstyle="round,pad=0.25"),
)

# de-emphasize the non-core execution time
axis_left.axvspan(
    xmin=datetime.fromtimestamp(job_start, tz=ZoneInfo("Europe/Berlin")),  # type: ignore
    xmax=datetime.fromtimestamp(core_start, tz=ZoneInfo("Europe/Berlin")),  # type: ignore
    hatch="X",
    facecolor="none",
    edgecolor="#dedede",
    linewidth=0.5,
)
axis_left.axvspan(
    xmin=datetime.fromtimestamp(core_end, tz=ZoneInfo("Europe/Berlin")),  # type: ignore
    xmax=datetime.fromtimestamp(job_end, tz=ZoneInfo("Europe/Berlin")),  # type: ignore
    hatch="X",
    facecolor="none",
    edgecolor="#dedede",
    linewidth=0.5,
)

# fmt: off
axis_left.autoscale(axis="x", tight=True)
axis_right.autoscale(axis="y", tight=True)
axis_left.set_ylim(bottom=0, top=power_kilowatt_limit)
# automatically adjust y axis limit to force a tick exactly at the axis limit
while axis_left.get_yticks()[-1] != axis_left.get_ylim()[1]:
    axis_left.set_ylim(top=axis_left.get_ylim()[1] + 1)
label_left = axis_left.set_ylabel("Power Consumption (in kW)", rotation="horizontal", va="bottom", y=1.02, ha="left")
label_right = axis_right.set_ylabel("Energy Consumption (in kWh)", rotation="horizontal", va="bottom", y=1.02, ha="right")
axis_left.xaxis.set_major_formatter(mdates.DateFormatter("%d.%b\n%H:%M"))
axis_left.grid(color="#dedede", linewidth=0.5)
# fmt: on

# match the number of primary and secondary y axis ticks to align the grid
# make the total energy consumption directly visible on the y axis
axis_right.yaxis.set_major_locator(mticker.LinearLocator(numticks=len(axis_left.get_yticks())))
# right aligning secondary y axis tick labels requires manual shifting of all labels
widest_yticklabel = max(axis_right.get_yticklabels(), key=lambda item: item.get_window_extent().width)
widest_bbox = widest_yticklabel.get_transform().inverted().transform(widest_yticklabel.get_window_extent())
min_x, max_x = widest_bbox[:, 0]
shifted_x = 1 + (max_x - min_x)
for tick_label in axis_right.get_yticklabels():
    tick_label.set_horizontalalignment("right")
    tick_label.set_x(shifted_x)

plot_left.set_color("tab:blue")
axis_right.spines["left"].set_color("tab:blue")
axis_left.tick_params(axis="y", colors="tab:blue")
label_left.set_color("tab:blue")

plot_right.set_color("tab:orange")
axis_right.spines["right"].set_color("tab:orange")
axis_right.tick_params(axis="y", colors="tab:orange")
label_right.set_color("tab:orange")

figure.savefig(output_directory_path / "summary.pdf", bbox_inches="tight")

output_file.close()
