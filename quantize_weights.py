import tensorflow.compat.v1 as tf
import io
import PIL
import numpy as np
import argparse
import sys


parser = argparse.ArgumentParser("qunatize_weights")

parser.add_argument('--sample_data', help='Sample data to estimate the min/max range', required=True)
parser.add_argument('--frozen_model', help='Path to frozen model', required=True)

args = parser.parse_args()

sample_dataset_path = args.sample_data
frozen_model = args.frozen_model

def representative_dataset_gen():
  record_iterator = tf.python_io.tf_record_iterator(path=sample_dataset_path)
  count = 0
  for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)
    image_stream = io.BytesIO(example.features.feature['image/encoded'].bytes_list.value[0])
    image = PIL.Image.open(image_stream)
    image = image.resize((96, 96))
    image = image.convert('L')
    array = np.array(image)
    array = np.expand_dims(array, axis=2)
    array = np.expand_dims(array, axis=0)
    array = ((array / 127.5) - 1.0).astype(np.float32)
    yield([array])
    count += 1
    if count > 300:
        break

converter = tf.lite.TFLiteConverter.from_frozen_graph(frozen_model, ['input'], ['MobilenetV1/Predictions/Reshape_1'])
converter.inference_input_type = tf.lite.constants.INT8
converter.inference_output_type = tf.lite.constants.INT8
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen

tflite_quant_model = converter.convert()
open("person_detection.tflite", "wb").write(tflite_quant_model)



