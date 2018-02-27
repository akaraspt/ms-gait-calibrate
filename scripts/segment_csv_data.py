import argparse
import datetime
import os
import shutil

import pandas as pd

from gaitcalibrate import dt_format
from gaitcalibrate.util.generate_datetime import hour_range
from gaitcalibrate.util.segment import segment_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True,
                       help="File path to the CSV file.")
    parser.add_argument("--sampling_rate", type=float, default=100.0,
                       help="Sampling rate used to record this CSV file.")
    parser.add_argument("--body_location", type=str, default="lower_back",
                       help="Device location on the human body.")
    parser.add_argument("--position", type=str, default="center_right",
                       help="Position of the device.")
    parser.add_argument("--output_dir", type=str, default='outputs/segments',
                       help="Directory where to save outputs.")
    args = parser.parse_args()

    # Output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    segment_csv(
        csv_file=args.csv_file,
        sampling_rate=args.sampling_rate,
        body_location=args.body_location,
        position=args.position,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
