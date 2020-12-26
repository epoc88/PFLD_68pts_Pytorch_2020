# Load ONNX model and convert to TensorFlow format
model_onnx = onnx.load('./models/model_simple.onnx')

tf_rep = prepare(model_onnx)

# Export model as .pb file
tf_rep.export_graph('./models/model_simple.pb')