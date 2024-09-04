# Create TFRecord dataset V3, Randomizing Records

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Modified by: Felipe Alcalde (falcalde21@alumno.uned.es)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Creates training and eval data from Quickdraw NDJSON files.

This tool reads the NDJSON files from https://quickdraw.withgoogle.com/data
and converts them into tensorflow.Example stored in TFRecord files.

The tensorflow example will contain 3 features:
 shape - contains the shape of the sequence [length, dim] where dim=3.
 class_index - the class index of the class for the example.
 ink - a length * dim vector of the ink.

It creates disjoint training and evaluation sets.

Script modified for TF v2.x and Python 3 compatibility and
to ensure that the records stored in the TFRecord shards come from
a randomized selection across all .ndjson files
"""

import json
import numpy as np
import os
import random
import tensorflow as tf

OUTPUT_SHARDS = 100 # Number of shards for the output.
OUTPUT_FILE = 'dataset/'
OBSERVATIONS_PER_CLASS = 88000  # How many items per class to load for training
EVAL_OBSERVATIONS_PER_CLASS = 12000  # How many items per class to load for evaluation

os.makedirs(OUTPUT_FILE, exist_ok=True) # Create the TFRecord directory

def parse_line(ndjson_line):
    try:
        sample = json.loads(ndjson_line)
    except json.JSONDecodeError:
        print("Invalid JSON format")
        return None, None
    class_name = sample.get("word", "")
    if not class_name:
        print("Empty classname")
        return None, None
    inkarray = sample.get("drawing", [])
    if not inkarray:
        print("Empty inkarray")
        return None, None
    stroke_lengths = [len(stroke[0]) for stroke in inkarray]
    total_points = sum(stroke_lengths)
    np_ink = np.zeros((total_points, 3), dtype=np.float32)
    current_t = 0
    for stroke in inkarray:
        if len(stroke[0]) != len(stroke[1]):
            print("Inconsistent number of x and y coordinates.")
            return None, None
        stroke_len = len(stroke[0])
        np_ink[current_t:current_t + stroke_len, :2] = np.transpose(stroke[:2])
        np_ink[current_t + stroke_len - 1, 2] = 1  # stroke_end
        current_t += stroke_len
    # Preprocessing
    # 1. Size normalization
    lower = np.min(np_ink[:, 0:2], axis=0)
    upper = np.max(np_ink[:, 0:2], axis=0)
    scale = np.where(upper - lower == 0, 1, upper - lower)
    np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale
    # 2. Compute deltas
    np_ink[1:, 0:2] -= np_ink[:-1, 0:2]
    np_ink = np_ink[1:, :]
    return np_ink, class_name

def get_file_line_counts(trainingdata_dir):
    """Counts the number of lines in each file and returns a list of tuples (filename, line_count)."""
    file_line_counts = []
    for filename in os.listdir(trainingdata_dir):
        if filename.endswith(".ndjson"):
            file_path = os.path.join(trainingdata_dir, filename)
            with open(file_path, 'r') as file:
                line_count = sum(1 for _ in file)
                file_line_counts.append((filename, line_count))
    return file_line_counts

def generate_global_random_indices(file_line_counts, total_samples, offset=0):
    """Generates a list of unique random indices across all files, considering an offset."""
    global_indices = []
    for _ in range(total_samples):
        file_index = random.randrange(len(file_line_counts))
        line_index = random.randrange(offset, file_line_counts[file_index][1])
        global_indices.append((file_index, line_index))
    return global_indices

def convert_data(trainingdata_dir, observations_per_class, output_file, classnames, output_shards=10, offset=0):
    os.makedirs(output_file, exist_ok=True) # Ensure the output directory exists

    def _pick_output_shard():
        """Selects a random shard to write the example."""
        return random.randint(0, output_shards - 1)

    # Initialize TFRecord writers for each shard
    writers = []
    for i in range(output_shards):
        filename = os.path.join(output_file, f"output-{i:05d}-of-{output_shards:05d}.tfrecord")
        writers.append(tf.io.TFRecordWriter(filename))

    file_line_counts = get_file_line_counts(trainingdata_dir)
    global_indices = generate_global_random_indices(file_line_counts, observations_per_class, offset)

    # Open all files for reading
    file_handles = [open(os.path.join(trainingdata_dir, f[0]), 'r') for f in file_line_counts]
    record_counter = 0

    for file_index, line_index in global_indices:
        file_handles[file_index].seek(0)  # Reset file pointer to start
        for _ in range(line_index):
            next(file_handles[file_index])  # Skip to the selected line
        line = file_handles[file_index].readline()
        ink, class_name = parse_line(line)
        if ink is not None and class_name in classnames:
            # Prepare TensorFlow example
            features = {
                "class_index": tf.train.Feature(int64_list=tf.train.Int64List(value=[classnames.index(class_name)])),
                "ink": tf.train.Feature(float_list=tf.train.FloatList(value=ink.flatten())),
                "shape": tf.train.Feature(int64_list=tf.train.Int64List(value=ink.shape))
            }
            f = tf.train.Features(feature=features)
            example = tf.train.Example(features=f)
            shard_index = _pick_output_shard()
            writers[shard_index].write(example.SerializeToString())

            # Print information about the record
            record_counter += 1
            print(f"Record #{record_counter} (original line #{line_index}) taken from file '{file_line_counts[file_index][0]}', stored in shard #{shard_index}")


    # Close all file handles and writers
    for w in writers:
        w.close()
    for f in file_handles:
        f.close()

    # Write the class list to a file
    with tf.io.gfile.GFile(os.path.join(OUTPUT_FILE, ".classes"), "w") as f:
        for class_name in classnames:
            f.write(class_name + "\n")

#Create TFrecords for training
convert_data(
    trainingdata_dir=file_path,
    observations_per_class=OBSERVATIONS_PER_CLASS,
    output_file=os.path.join(OUTPUT_FILE, "training.tfrecord"),
    classnames=categories,
    output_shards=OUTPUT_SHARDS,
    offset=0
)
#Create TFrecords for evaluation
convert_data(
    trainingdata_dir=file_path,
    observations_per_class=EVAL_OBSERVATIONS_PER_CLASS,
    output_file=os.path.join(OUTPUT_FILE, "eval.tfrecord"),
    classnames=categories,
    output_shards=OUTPUT_SHARDS,
    offset=OBSERVATIONS_PER_CLASS
)