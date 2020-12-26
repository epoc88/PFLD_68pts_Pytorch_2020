import torch
import onnx


model_pytorch = SimpleModel(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
model_pytorch.load_state_dict(torch.load('./mobileNetV2_0.25.pth'))

dummy_input = torch.from_numpy(X_test[0].reshape(1, -1)).float().to(device)
dummy_output = model_pytorch(dummy_input)
print(dummy_output)

# Export to ONNX format
torch.onnx.export(model_pytorch, dummy_input, './mobileNetV2_0.25.onnx', input_names=['test_input'], output_names=['test_output'])
