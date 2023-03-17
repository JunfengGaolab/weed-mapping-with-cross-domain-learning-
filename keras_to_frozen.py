import _init_paths
import tensorflow as tf
from keras.models import model_from_json, load_model
from keras import backend as K
import os
from DeepLabv3 import relu6, BilinearUpsampling
from keras.utils.generic_utils import get_custom_objects
from data_loader_test import weighted_categorical_crossentropy
import numpy as np
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib

# weights1 = np.array([1, 1.5, 1.5])
# json_file = open('./save_trained_models/UNET_model.json', 'r')
# model_json = json_file.read()
# json_file.close()
# model = model_from_json(model_json, custom_objects={
#     'loss': weighted_categorical_crossentropy(weights1)})
# # load weights
# # model = UNET(3, 512, 512)
# model.load_weights('./save_trained_models/weed_maize_UNET_weighted3_test_2765.h5')
model = load_model('./save_trained_models/weed_maize_DeepLabv3_weighted_988.h5',
                   custom_objects={'relu6': relu6, 'BilinearUpsampling': BilinearUpsampling})
# rename output nodes
output_node_prefix = 'output_node_'
output_node_names = []
output_nodes = []
for i in range(len(model.outputs)):
    output_node_name = output_node_prefix + str(i)
    output_nodes.append(tf.identity(model.outputs[i], name=output_node_name))
    output_node_names.append(output_node_name)
    print(i)
print('output node names are: ', output_node_names)
# freeze the graph
sess = K.get_session()
constant_graph = tf.graph_util.convert_variables_to_constants(sess=sess,
                                                              input_graph_def=sess.graph.as_graph_def(),
                                                              output_node_names=output_node_names)
# write graph as pb
output_dir = 'frozen_graph'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
filename = 'model.pb'
tf.train.write_graph(constant_graph, output_dir, filename, as_text=False)


# def convert_keras_to_pb(out_names, models_dir, model_filename):
#
#     model = load_model('./save_trained_models/weed_maize_UNET_988.h5')
#     K.set_learning_phase(0)
#     sess = K.get_session()
#     saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
#     checkpoint_path = saver.save(sess, 'saved_ckpt', global_step=0, latest_filename='checkpoint_state')
#     graph_io.write_graph(sess.graph, '.', 'tmp.pb')
#     freeze_graph.freeze_graph('./tmp.pb', '',
#                             False, checkpoint_path, out_names,
#                             "save/restore_all", "save/Const:0",
#                             models_dir + model_filename, False, "")
#
#
#
# convert_keras_to_pb('unet', './frozen_graph/', 'unetpb')


