# GPT-2 Text Generation Console Application

This C# console application uses a GPT-2 model in ONNX format for text generation based on user input. It predicts tokens one by one to generate coherent text, utilizing the GPT-2 model fine-tuned for various language generation tasks.

## Features

- **GPT-2 Model**: Utilizes a pre-trained GPT-2 Large model (`gpt2-large.onnx`) for generating predictions.
- **ONNX Runtime**: The model is loaded and executed using ONNX Runtime for inference.
- **Dynamic Model Download**: If the model does not exist locally, it is automatically downloaded and stored.
- **Custom Tokenization**: The application uses GPT-2’s tokenizer to convert input text into tokens and decode output tokens back into text.
- **Python Integration**: PythonExecutor is used to manage Python dependencies for exporting ONNX models.

## Installation

### Prerequisites

- .NET 8.0 or higher
- Python 3.7 or higher (for PythonExecutor)
- ONNX Runtime

### NuGet Packages

The following NuGet packages are required for the project:

xml
```
<PackageReference Include="Microsoft.ML.Tokenizers" Version="3.0.1" />
<PackageReference Include="Microsoft.ML.OnnxTransformer" Version="3.0.1" />
<PackageReference Include="Python.Included" Version="3.11.6" />
```
## Setup
Clone this repository to your local machine:

git clone https://github.com/xx4g/Dot-Net-Gpt2-Ai-Chatbot.git
cd BDot-Net-Gpt2-Ai-Chatbot
Ensure you have the necessary .NET and Python dependencies installed.

Build the project using .NET CLI:

bash

```
dotnet build
```
##Usage
Run the application:

bash

```
dotnet run
```
The application starts a loop asking for user input and returns the answer using the gpt2-large model.

Exit the loop by typing exit.

Example:
```
Enter your input query:
What is the tallest building in the world?
The tallest building in the world is the Burj Khalifa in Dubai, which is a whopping 1,716 feet tall.
Enter your input query:
What is the capital of the united states?
The capital of the United States is Washington, D.C.

```
![image](https://github.com/user-attachments/assets/37d6bd66-ca20-4c5e-a576-51421520d9f6)




## Model Details
The application utilizes the gpt2-large model, which is used for text generation tasks. The model is downloaded if not found in the models directory.

## Code Overview

```
// See https://aka.ms/new-console-template for more information
using GptNlp;
using Dot_Net_Gpt2_Ai_Chatbot.PythonExecutor;
using Nexez;


class Program
{
    static internal async Task Main(string[] args)
    {

        // Define the path for the "models" folder within the current working directory
        string modelsPath = Path.Combine(Directory.GetCurrentDirectory(), "models\\gpt2-large");
        string modelPath = System.IO.Path.Combine(modelsPath, "gpt2-large.onnx");

        // Check if the "models" directory already exists
        if (!File.Exists(modelPath))
        {
            // If it does not exist, create it
            Directory.CreateDirectory(modelsPath);
            Console.WriteLine($"Created directory at: {modelsPath}");

            // Initialize PythonExecutor asynchronously
            PythonExecutor pythonExecutor = new PythonExecutor(new[] { "transformers", "numpy", "torch", "onnx", "onnxruntime" });

            try
            {
                await pythonExecutor.Initialize(); // Non-blocking, awaiting the initialization
                Console.WriteLine("PythonExecutor initialized successfully.");

                // Execute Python script to export ONNX model
                pythonExecutor.ExecuteScript(Dot_Net_Gpt2_Ai_Chatbot.PythonExportOnnxModel.pythonScript);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"PythonExecutor initialization failed: {ex.Message}");
            }
            finally
            {
                pythonExecutor.Shutdown();
            }
        }
        else
        {
            Console.WriteLine($"Model already exists at: {modelPath}");
        }
        if (!File.Exists(modelPath))
        {
            Console.WriteLine($"Model not found at: {modelPath}");
            return;
        }
        var myinferenceSession = OnnxInferenceSession.GetInstance(modelPath);
        if (myinferenceSession == null)
        {
            Console.WriteLine($"Model null at: {modelPath}");
            return;
        }

        while (true)
        {
            // Prompt user for input
            Console.WriteLine("Enter your input query:");
            string userInput = Console.ReadLine();
            if (userInput == "exit")
            {
                break;
            }
            var bestTokens = myinferenceSession.GetResponse($"{userInput} ");
           var trimmed = StringTrimmer.TrimPrefix(StringTrimmer.TrimSuffix(bestTokens.ToString().Trim()));
            // Display the best response with the highest probability
            Console.WriteLine($"{trimmed}");
        }
    }
}

```

## License

```
This project is licensed under the MIT License. See the LICENSE file for more details.
```
