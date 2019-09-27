import tensorflow as tf
import tensorflow.contrib.keras as K
tf.keras.backend.set_learning_phase(0)



def get_op_name(outputs,graph):
    for operation in graph.get_operations():
        if operation.outputs == outputs:
            return operation.name



def save_model(model,graph_file,optimize=True):
    sess = K.backend.get_session()
    graph = sess.graph
    input_name = get_op_name(model.inputs,graph)
    out_name = get_op_name(model.outputs, graph)
    graph_def = graph.as_graph_def()
    constant_graph = tf.graph_util.convert_variables_to_constants(sess,graph_def, [out_name])
    tf.train.write_graph(constant_graph, "", graph_file, as_text=False)
    if optimize:
        optimize_for_inference(graph_file,input_name,out_name)

import os
tf_path = "/home/user/Workspace/tensorflow"

def optimize_for_inference(graph_file,input_name,out_name):
    command = f"python {tf_path}/tensorflow/python/tools/optimize_for_inference.py \
      --input {graph_file} \
      --output {graph_file} \
      --frozen_graph True \
      --input_names {input_name} \
      --output_names {out_name}"
    res = os.popen(command).read()
    print(res)


if __name__ == '__main__':
    model = K.applications.MobileNet()
    model.layers
    save_model(model,"mobilenet.pb")
    import cv2
    net = cv2.dnn.readNetFromTensorflow('mobilenet.pb')
