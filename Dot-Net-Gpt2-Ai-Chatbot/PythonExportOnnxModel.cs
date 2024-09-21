using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dot_Net_Gpt2_Ai_Chatbot
{
    internal class PythonExportOnnxModel
    {
      static internal string pythonScript = @"
import os
import torch
import json
import onnxruntime as ort
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import sys

# Function to create the models folder
def create_models_folder():
    models_path = os.path.join(os.path.join(os.getcwd(), 'models'), 'gpt2-large')
    if not os.path.exists(models_path):
        os.makedirs(models_path)
        print(f'Created directory at: {models_path}')
    else:
        print(f'Directory already exists at: {models_path}')
    return models_path

# Function to export the model to ONNX format
def export_model_to_onnx(model, tokenizer, models_folder_path):
    model.eval()
    dummy_input = torch.tensor([tokenizer.encode('Hello, what time is it?', add_special_tokens=True)]).long()
    onnx_model_path = models_folder_path
    torch.onnx.export(model, dummy_input, onnx_model_path, export_params=True, opset_version=14, 
                      do_constant_folding=True, input_names=['input_ids'], 
                      output_names=['logits'], 
                      dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'}, 
                                    'logits': {0: 'batch_size', 1: 'sequence_length'}})
    print(f'Model has been exported to ONNX format at {onnx_model_path}')
    return onnx_model_path

# Print statements within this block will be redirected to console_output.txt
print(f'Creating models folder...')
models_folder_path = create_models_folder()

print(f'GPT2Tokenizer')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large', cache_dir=models_folder_path)
tokenizer.pad_token = tokenizer.eos_token

print(f'GPT2LMHeadModel')
model = GPT2LMHeadModel.from_pretrained('gpt2-large', cache_dir=models_folder_path)

onnx_model_filename = 'gpt2-large.onnx'
onnx_model_path = os.path.join(models_folder_path, onnx_model_filename)
print(f'ONNX model path: {onnx_model_path}')
export_model_to_onnx(model, tokenizer, onnx_model_path)

print(f'Finished Exporting Onnx Model...')
";

    }
}
