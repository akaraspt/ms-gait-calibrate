import argparse
import pandas as pd

from gaitcalibrate import dt_format


def reverse_upside_down(in_filepath, out_filepath):
    """From this orientation <[rotate_180(A)]> to this <[A]>."""

    def dateparse(x): return pd.datetime.strptime(x, dt_format)
    
    # Read the input CSV file
    acc = pd.read_csv(
        in_filepath,
        names=['dt', 'x', 'y', 'z'],
        header=None,
        parse_dates=['dt'],
        date_parser=dateparse
    )

    acc['x'] = -acc['x']
    acc['y'] = -acc['y']

    # Save the reversed CSV file
    acc.to_csv(
        out_filepath,
        header=False,
        index=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv_file", type=str, required=True,
                       help="File path to the input CSV file.")
    parser.add_argument("--output_csv_file", type=str, required=True,
                       help="File path to save the output CSV file.")
    args = parser.parse_args()

    reverse_upside_down(
        in_filepath=args.input_csv_file,
        out_filepath=args.output_csv_file
    )


if __name__ == "__main__":
    main()