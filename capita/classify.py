import sys
import tensorflow as tf

config = tf.ConfigProto(
    device_count={'GPU': 0}
)
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.system('cls' if os.name == 'nt' else 'clear')
tf.logging.set_verbosity(tf.logging.FATAL)

# change this as you see fit
from tensorflow.python.framework.errors_impl import NotFoundError


def tensorflow_run(image_path):
    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("retrained_labels.txt")]

    with tf.Session(config=config) as sess:

        saver = tf.train.import_meta_graph('./faces_model.ckpt.meta')
        # Feed the image_data as input to the graph and get first prediction
        try:
            saver.restore(sess, './faces_model.ckpt')
            print("#################### USING SAVED MODEL TO CLASSIFY ################################")
        except (IOError, NotFoundError) as e:
            print("################### NO SAVED MODEL CANNOT CLASSIFY###################")
            sys.exit()

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s = %s (score = %.5f)\n' % (image_path.split('.')[0], human_string, score))
        print(label_lines[top_k[0]])
        return label_lines[top_k[0]]


tensorflow_run(sys.argv[1])
