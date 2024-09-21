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
