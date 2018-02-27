import datetime
import ntpath
import os

import pandas as pd

from gaitcalibrate import dt_format
from gaitcalibrate.util.generate_datetime import hour_range


def segment_csv(
    csv_file,
    sampling_rate,
    body_location,
    position,
    output_dir
):
    filename = ntpath.basename(csv_file)
    filename = filename.split(".")[0]

    # The number of rows in the file to be read for each iteration (or chunk)
    chunksize = 50000

    # Number of rows to skip from the start of the file
    skiprows = 0

    # Function to parse datetime format
    def dateparse(x): return pd.datetime.strptime(x, dt_format)

    # Get chunk iterator
    chunk_iterator = pd.read_csv(
        csv_file,
        names=['dt', 'x', 'y', 'z'],
        header=None,
        parse_dates=['dt'],
        date_parser=dateparse,
        skiprows=skiprows,
        chunksize=chunksize
    )

    # Read the first line to get the start datetime
    with open(csv_file, 'r') as f:
        first_line = f.readline()
        cols = first_line.split(",")
        start_dt = pd.datetime.strptime(cols[0], dt_format)

    # Iterate through each hour
    list_chunks = []
    seg_idx = 1
    last_segment = False
    for end_dt in hour_range(start_dt=start_dt, 
                             end_dt=start_dt+datetime.timedelta(days=10),
                             include_start=False):
        print "[{}] Segment {}: {} - {}".format(filename, seg_idx, start_dt, end_dt)

        while(True):
            try:
                # Get one chunk from the CSV file
                acc_chunk = chunk_iterator.get_chunk()
            except:
                # Reach the last chunk
                if len(list_chunks) > 0:
                    hour_chunk = pd.concat(list_chunks)
                    # Add metadata at the first line of the file
                    output_filepath = os.path.join(output_dir, "{}.csv".format(seg_idx))
                    with open(output_filepath, 'w') as f:
                        f.write("{},{},{}\n".format(
                            sampling_rate,
                            body_location,
                            position
                        ))
                    # Append the acceleration data
                    with open(output_filepath, 'a') as f:
                        hour_chunk.to_csv(
                            f,
                            header=False,
                            index=False
                        )
                    del list_chunks[:]
                    last_segment = True
                break

            # Get start and end datetime of the chunk
            start_chunk_dt = acc_chunk['dt'].values[0]
            end_chunk_dt = acc_chunk['dt'].values[-1]

            # # Convert to datetime in python
            # start_chunk_dt = pd.to_datetime(acc_chunk['dt'].values[0])
            # end_chunk_dt = pd.to_datetime(acc_chunk['dt'].values[-1])

            if pd.to_datetime(end_chunk_dt) >= end_dt:
                tmp_chunk = acc_chunk[acc_chunk['dt'] <= end_dt]
                list_chunks.append(tmp_chunk)
                hour_chunk = pd.concat(list_chunks)
                # Add metadata at the first line of the file
                output_filepath = os.path.join(output_dir, "{}.csv".format(seg_idx))
                with open(output_filepath, 'w') as f:
                    f.write("{},{},{}\n".format(
                        sampling_rate,
                        body_location,
                        position
                    ))
                # Append the acceleration data
                with open(output_filepath, 'a') as f:
                    hour_chunk.to_csv(
                        f,
                        header=False,
                        index=False
                    )
                del list_chunks[:]
                seg_idx += 1

                # Add the small chunk of the next hour into the list
                list_chunks = [acc_chunk[acc_chunk['dt'] > end_dt]]
                break
            else:
                list_chunks.append(acc_chunk)
            
        # Move to the next hour
        start_dt = end_dt + datetime.timedelta(milliseconds=1)

        # Stop
        if last_segment:
            break

    chunk_iterator.close()