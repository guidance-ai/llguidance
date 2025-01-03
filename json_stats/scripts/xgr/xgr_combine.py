#!/usr/bin/env python3

import json
import math
import glob

output_path = "tmp/xgr/"
llg = False


class Stats:
    def __init__(self) -> None:
        self.ttfm_us = 0
        self.max_ttfm_us = 0
        self.masks_us = 0
        self.masks_us_under_10ms = 0
        self.num_masks_under_10ms = 0
        self.avg_masks_under_10ms = 0
        self.masks_us_over_10ms = 0
        self.num_masks_over_10ms = 0
        self.avg_masks_over_10ms = 0
        self.max_mask_us = 0
        self.num_tokens = 0
        self.num_schemas = 0
        self.num_schemas_ok = 0
        self.num_compilation_errors = 0
        self.num_validation_errors = 0
        self.num_tests = 0
        self.num_valid_tests = 0
        self.num_invalid_tests = 0


def log_fraction_plot(times: list[int]):
    times.sort()
    cutoff = 1
    mult = 1.3
    count = 0
    csv = "cutoff time,frac left\n"
    total = len(times)
    for t in times:
        while t > cutoff:
            csv += f"{cutoff/1000.0},{(total - count)/total}\n"
            cutoff = int(cutoff * mult) + 1
        count += 1
    return csv


def histogram_position(us: int):
    return int(math.floor(math.log10(max(1, us - 1))))

def us_to_str(us: int):
    if us < 1000:
        return f"{us}us"
    if us < 1000000:
        return f"{us//1000}ms"
    return f"{us//1000000}s"


def main():
    if llg:
        files = ["tmp/llg_results.json"]
    else:
        files = glob.glob(output_path + "*.json")
        files = sorted(files)
    stats = Stats()
    ttfm_us = []
    all_masks_us = []
    histogram_us = [0] * 10
    histogram_num = [0] * 10
    for f in files:
        with open(f) as f:
            data = json.load(f)
        elts = [data]
        if isinstance(data, list):
            elts = data
        for data in elts:
            if "num_tests" not in data:
                continue
            stats.num_schemas += 1
            stats.num_tests += data["num_tests"]
            if "compile_error" in data:
                stats.num_compilation_errors += 1
            else:
                stats.ttfm_us += data["ttfm_us"]
                ttfm_us.append(data["ttfm_us"])
                stats.max_ttfm_us = max(data["max_ttfm_us"], stats.max_ttfm_us)
                if "masks_us" in data:
                    stats.masks_us += data["masks_us"]
                    stats.max_mask_us = max(data["max_mask_us"], stats.max_mask_us)
                    stats.num_tokens += data["num_tokens"]
                if "validation_error" in data:
                    stats.num_validation_errors += 1
                else:
                    stats.num_schemas_ok += 1
                stats.num_valid_tests += data["num_valid_tests"]
                stats.num_invalid_tests += data["num_invalid_tests"]
                all_masks_us.extend(data["all_mask_us"])
                for us in data["all_mask_us"]:
                    p = histogram_position(us)
                    histogram_us[p] += us
                    histogram_num[p] += 1
                    if us < 10000:
                        stats.masks_us_under_10ms += us
                        stats.num_masks_under_10ms += 1
                    else:
                        stats.masks_us_over_10ms += us
                        stats.num_masks_over_10ms += 1
    stats.avg_masks_under_10ms = stats.masks_us_under_10ms // stats.num_masks_under_10ms
    stats.avg_masks_over_10ms = stats.masks_us_over_10ms // stats.num_masks_over_10ms
    print(json.dumps(stats.__dict__, indent=2))
    with open("tmp/xgr_ttfm_us.csv", "w") as f:
        f.write(log_fraction_plot(ttfm_us))
    with open("tmp/xgr_masks_us.csv", "w") as f:
        f.write(log_fraction_plot(all_masks_us))
    
    all_masks_us.sort()
    for p in [25, 50, 75, 90, 95, 99, 99.9, 99.99]:
        print(f"{'p' + str(p):7}, {all_masks_us[int(len(all_masks_us) * p / 100)]}")

    num_masks = sum(histogram_num)
    h_csv = "above us,frac\n"
    for i in range(10)[1:]:
        frac = sum(histogram_num[i:]) * 100 / num_masks
        h_csv += f"{us_to_str(10**i):10}"
        h_csv += f","
        h_csv += f"{frac:1.15}"
        h_csv += f"\n"
    with open("tmp/xgr_histogram.csv", "w") as f:
        f.write(h_csv)
    print(h_csv)


main()
