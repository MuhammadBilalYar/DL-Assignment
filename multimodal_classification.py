# run following command in terminal to install bert
# !pip install bert-for-tf2

# run file using the following command from terminal
# python3 multimodal_classification.py --train_path <path> --val_path <path> \
# --test_path <path> --image_folder <folder_path>

import os
import sys
import warnings
sys.path.append("C:\ProgramData\Anaconda3")
sys.path.append("C:\ProgramData\Anaconda3\Lib\site-packages")
sys.path.append("C:\ProgramData\Anaconda3\Lib\site-packages\win32")
sys.path.append("C:\ProgramData\Anaconda3\Lib\site-packages\win32\lib")
sys.path.append("F:\Download\Dataset")
import argparse
import bert
import numpy as np
from nltk.corpus import stopwords
import tensorflow as tf                 # tensorflow version 2.4 (>=2 required)
import tensorflow_hub as hub
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import VGG16
from sklearn.metrics import f1_score, accuracy_score
from Code.jpeg_decoder import JPEG
stop = stopwords.words("english")
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# parse arguments
# argparser = argparse.ArgumentParser()
# argparser.add_argument("-t", "--train_path", required=True)
# argparser.add_argument("-v", "--val_path", required=True)
# argparser.add_argument("-e", "--test_path", required=True)
# argparser.add_argument("-i", "--image_folder", required=True)
# args = vars(argparser.parse_args())

train_path = "F:\Download\Dataset\multimodal_only_samples\multimodal_train.tsv"
val_path = "F:\Download\Dataset\multimodal_only_samples\multimodal_validate.tsv"
test_path = "F:\Download\Dataset\multimodal_only_samples\multimodal_test_public.tsv"
image_folder = "F:\Download\Dataset\public_image_set"

# initialize hyper-parameters
_max_len = 150
batch_size = 16
_epochs = 1
num_labels = 6
sample_count = 1000      # sample for testing functionality of code
# None for using all samples, or any integer

# function to load the small bert model and configure it


def load_bert_model():
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2",
                                trainable=False)
    bert_tokenizer = bert.bert_tokenization.FullTokenizer
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = bert_tokenizer(vocabulary_file, to_lower_case)
    return bert_layer, tokenizer


# function to check valid images
def check_image(file_path):
    img = tf.io.read_file(file_path)
    return tf.io.is_jpeg(img)


# second jpeg filter to check if the file is corrupt
def jpeg_decoder(ds):
    train_data_images = list(ds.map(lambda x: x["id"][0]).as_numpy_iterator())

    new_list = []
    for image in train_data_images:
        image = image.decode("utf-8")
        filepath = os.path.join(image_folder, image) + ".jpg"
        try:
            JPEG(filepath).decode()
        except:
            new_list.append(image)

    new_list = tf.convert_to_tensor(new_list)

    if len(new_list):
        new_ds = ds.filter(lambda x: tf.math.logical_not(tf.math.reduce_any(tf.math.equal(x["id"], new_list))))
        return new_ds
    else:
        return ds


# function to parse images
def parse_image(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])

    return img


# function to encode texts into bert embeddings
def bert_encode_modified(tokenizer, data, bert_layer, max_len, mode="sequence"):

    def convert_tokens(tokenizer, text, max_len):

        tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

        if len(tokens) <= max_len - 2:
            tokens.append(tokenizer.convert_tokens_to_ids(["[SEP]"])[0])

        else:
            tokens = tokens[:max_len - 2] + \
                tokenizer.convert_tokens_to_ids(["[SEP]"])

        return tokens

    sentence = tf.ragged.constant(
        [convert_tokens(tokenizer, s["clean_title"].numpy()[0], max_len) for s in data])
    cls_token = [tokenizer.convert_tokens_to_ids(
        ["[CLS]"])] * sentence.shape[0]
    input_word = tf.concat([cls_token, sentence], axis=-1)

    mask = tf.ones_like(input_word).to_tensor()
    type_id = tf.zeros_like(input_word).to_tensor()
    input_word = input_word.to_tensor()

    seq_length = input_word.shape[1]
    pad_length = max_len - seq_length

    if pad_length > 0:
        pad_token = [[0] * pad_length] * sentence.shape[0]
        input_word = tf.concat([input_word, pad_token], axis=-1)
        mask = tf.concat([mask, pad_token], axis=-1)
        type_id = tf.concat([type_id, pad_token], axis=-1)

    inputs = {'input_word_ids': input_word,
              'input_mask': mask,
              'input_type_ids': type_id}

    bert_embeddings = bert_layer(inputs)

    if mode == "pooled":
        return bert_embeddings["pooled_output"]

    else:
        return bert_embeddings["sequence_output"]


# function to build deep learning model
def classification_model(max_len, num_labels):
    # text model
    text_input = Input(shape=(max_len, 128,))
    layer = Bidirectional(LSTM(64, return_sequences=True))(text_input)
    layer = Bidirectional(LSTM(32))(layer)
    text_flat = Flatten()(layer)

    # image model
    vgg_16 = VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    vgg_16.trainable = False
    vgg_output = vgg_16.output
    image_flat = Flatten()(vgg_output)
    image_output = Dense(64, activation="relu")(image_flat)

    # concatenate the inputs to form a single flattened vector
    final_input = Concatenate(axis=1)([text_flat, image_output])
    dense = Dense(32, activation="relu")(final_input)
    output = Dense(num_labels, activation="softmax")(dense)

    # return final model
    model = Model(inputs=[text_input, vgg_16.input], outputs=output)
    return model


# read the text data into Tensorflow Data API
train_data = tf.data.experimental.make_csv_dataset(train_path, batch_size=1,
                                                   select_columns=[
                                                       "clean_title", "id", "image_url", "6_way_label"],
                                                   field_delim="\t", shuffle=False)

val_data = tf.data.experimental.make_csv_dataset(val_path, batch_size=1,
                                                 select_columns=[
                                                     "clean_title", "id", "image_url", "6_way_label"],
                                                 field_delim="\t", shuffle=False)

test_data = tf.data.experimental.make_csv_dataset(test_path, batch_size=1,
                                                  select_columns=[
                                                      "clean_title", "id", "image_url", "6_way_label"],
                                                  field_delim="\t", shuffle=False)

# filter na in image url column
if sample_count is not None:
    _skip = 20000
    train_data = train_data.skip(_skip).take(sample_count)
    val_data = val_data.skip(_skip).take(sample_count)
    test_data = test_data.skip(_skip).take(sample_count)

train_data = train_data.filter(lambda x: tf.math.reduce_all(
    tf.math.not_equal(x["image_url"], "")))
train_data = train_data.filter(lambda x: tf.math.logical_not(tf.strings.regex_full_match(x["id"][0], ".*[^\w]+.*")))
train_data = train_data.filter(lambda x: check_image(
    tf.strings.join([image_folder, "/", x["id"], ".jpg"])[0]))
train_data = jpeg_decoder(train_data)

val_data = val_data.filter(lambda x: tf.math.reduce_all(
    tf.math.not_equal(x["image_url"], "")))
val_data = val_data.filter(lambda x: tf.math.logical_not(tf.strings.regex_full_match(x["id"][0], ".*[^\w]+.*")))
val_data = val_data.filter(lambda x: check_image(
    tf.strings.join([image_folder, "/", x["id"], ".jpg"])[0]))
val_data = jpeg_decoder(val_data)

test_data = test_data.filter(lambda x: tf.math.reduce_all(
    tf.math.not_equal(x["image_url"], "")))
test_data = test_data.filter(lambda x: tf.math.logical_not(tf.strings.regex_full_match(x["id"][0], ".*[^\w]+.*")))
test_data = test_data.filter(lambda x: check_image(
    tf.strings.join([image_folder, "/", x["id"], ".jpg"])[0]))
test_data = jpeg_decoder(test_data)

# get images and their directories
train_images_path = train_data.map(lambda x: x["id"])
train_images_path = train_images_path.map(
    lambda x: tf.strings.join([image_folder, "/", x, ".jpg"])[0])
train_images = train_images_path.map(parse_image)
val_images_path = val_data.map(lambda x: x["id"])
val_images_path = val_images_path.map(
    lambda x: tf.strings.join([image_folder, "/", x, ".jpg"])[0])
val_images = val_images_path.map(parse_image)
test_images_path = test_data.map(lambda x: x["id"])
test_images_path = test_images_path.map(
    lambda x: tf.strings.join([image_folder, "/", x, ".jpg"])[0])
test_images = test_images_path.map(parse_image)


# get the labels and encode them into one hot vectors
train_labels = train_data.map(lambda x: x["6_way_label"])
train_labels = train_labels.map(lambda x: tf.one_hot(x, num_labels))
train_labels = train_labels.map(lambda x: tf.squeeze(x, axis=0))
val_labels = val_data.map(lambda x: x["6_way_label"])
val_labels = val_labels.map(lambda x: tf.one_hot(x, num_labels))
val_labels = val_labels.map(lambda x: tf.squeeze(x, axis=0))
test_labels = test_data.map(lambda x: x["6_way_label"])
# test_labels = test_labels.map(lambda x: tf.one_hot(x, num_labels))
# test_labels = test_labels.map(lambda x: tf.squeeze(x, axis=0))


# text embeddings from BERT and convert to TF Dataset format
bert_layer, tokenizer = load_bert_model()
train_embeddings = bert_encode_modified(tokenizer, train_data, bert_layer, max_len=_max_len, mode="sequence")
train_embeddings = tf.data.Dataset.from_tensor_slices(train_embeddings)
val_embeddings = bert_encode_modified(tokenizer, val_data, bert_layer, max_len=_max_len, mode="sequence")
val_embeddings = tf.data.Dataset.from_tensor_slices(val_embeddings)
test_embeddings = bert_encode_modified(tokenizer, test_data, bert_layer, max_len=_max_len, mode="sequence")
test_embeddings = tf.data.Dataset.from_tensor_slices(test_embeddings)

# zip the inputs and labels into a single dataset
train_dataset = tf.data.Dataset.zip(((train_embeddings, train_images), train_labels)).shuffle(
    100, reshuffle_each_iteration=False)
val_dataset = tf.data.Dataset.zip(((val_embeddings, val_images), val_labels)).shuffle(
    100, reshuffle_each_iteration=False)
test_dataset = tf.data.Dataset.zip(((test_embeddings, test_images), ))


# batch the datasets
train_ds = train_dataset.batch(batch_size)
val_ds = val_dataset.batch(batch_size)
test_ds = test_dataset.batch(batch_size)

# create a new model or load an existing model and compile it
model_folder = os.path.join(image_folder, "../", "models")

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

model_file = os.path.join(model_folder, "model.h5")

if os.path.exists(model_file):
    model = load_model(model_file)

else:
    model = classification_model(_max_len, num_labels)

model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics="accuracy")

# train the model
train = model.fit(train_ds, epochs=_epochs,
                  verbose=True, validation_data=val_ds)

# save the model to directory
model.save(model_file)

# predict on the test dataset
predictions = model.predict(test_ds)
predictions = np.argmax(predictions, axis=1)
true_labels = list(test_labels.as_numpy_iterator())

# classification metrics
print("Test Accuracy:", accuracy_score(true_labels, predictions))
print("Test F1 Score:", f1_score(true_labels, predictions, average=None))