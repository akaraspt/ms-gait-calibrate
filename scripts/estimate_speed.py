import argparse
import functools
import ntpath
import pickle
import os
import shutil

from multiprocessing.dummy import Pool as ThreadPool

import numpy as np

from gaitcalibrate.extract.walk import extract_walk_csv
from gaitcalibrate.estimate.speed import estimate_walk_speed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True,
                        help="File path to the CSV or NPY file that contains walking data.")
    parser.add_argument("--model_file", type=str, required=True,
                        help="File path to the trained model used to estimate walking speeds.")
    parser.add_argument("--output_dir", type=str, default="outputs/speeds",
                        help="Directory where to save outputs.")
    args = parser.parse_args()

    # Get input filename
    filename = ntpath.basename(args.input_file)
    filename = filename.split(".")[0]

    # Output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    # Load model
    if os.path.isfile(args.model_file):
        with open(args.model_file, 'rb') as f:
            model = pickle.load(f)
            sampling_rate = model['sampling_rate']
            body_location = model['body_location']
            position = model['position']
    else:
        raise Exception("Invalid model_file.")

    # Process input file
    if os.path.isfile(args.input_file):
        # CSV file
        if args.input_file.lower().endswith('.csv'):
            walks = extract_walk_csv(
                csv_file=args.input_file,
                n_skip_edge_step=2,
                thd_n_step_each_walk=10,
                sampling_rate=sampling_rate,
                body_location=body_location,
                position=position
            )
        # NPY file - contain a list of walks
        elif args.input_file.lower().endswith('.npy'):
            walks = np.load(args.input_file)
        else:
            raise Exception("Invalid input file.")

        if len(walks) == 0:
            print "There is no walk to estimate."
            return

        # Worker pool for multithreading
        pool = ThreadPool(8)

        # Specify arguments for estimate_walk_speed
        func = functools.partial(
            estimate_walk_speed,
            model=model,
            g2acc=True,
            n_skip_edge_step=2,
            thd_n_step_each_walk=10
        )

        # Estimate speed for each walk
        # Parallel map call - it blocks until the result is ready.
        outputs = pool.map(func, walks)

        # Save to an NPY file
        outputs = np.asarray(outputs)
        np.save(
            os.path.join(args.output_dir, "{}.npy".format(filename)),
            outputs
        )

    else:
        raise Exception("Invalid input_file.")


if __name__ == "__main__":
    main()
