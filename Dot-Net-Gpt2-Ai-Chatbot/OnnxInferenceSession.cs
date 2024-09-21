namespace GptNlp
{
    using Microsoft.ML.Tokenizers;
    using Microsoft.ML.OnnxRuntime;
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    /// <summary>
    /// Singleton class for managing ONNX inference sessions.
    /// </summary>
    internal class OnnxInferenceSession
    {
        /// <summary>
        /// Gets a singleton instance of OnnxInferenceSession for the specified model path.
        /// </summary>
        /// <param name="modelPath">The path to the ONNX model file.</param>
        /// <returns>An instance of OnnxInferenceSession.</returns>
        internal static OnnxInferenceSession GetInstance(string modelPath)
        {
            var instance = new OnnxInferenceSession(modelPath);
            return instance;
        }

        public OnnxInferenceSession(string path)
        {
            var vocabFilePath = "Assets\\Vocabularies\\vocab.json";
            var mergeFilePath = "Assets\\Vocabularies\\Merges.txt";
            inferenceSession = new Microsoft.ML.OnnxRuntime.InferenceSession(path);
            _tokenizer = new Microsoft.ML.Tokenizers.Tokenizer(new Bpe(vocabFilePath, mergeFilePath), RobertaPreTokenizer.Instance);
        }
        /// <summary>
        /// Gets the response from the GPT-style inference session for the given input text.
        /// </summary>
        /// <param name="text">The input text for which the response is generated.</param>
        /// <returns>A StringBuilder containing the generated response.</returns>
        /// <summary>
        /// Gets the response from the GPT-style inference session for the given input text.
        /// </summary>
        /// <param name="text">The input text for which the response is generated.</param>
        /// <returns>A StringBuilder containing the generated response.</returns>
        internal string GetResponse(string text)
        {
            try
            {
                StringBuilder filteredString = new StringBuilder();
                var encoded = _tokenizer.Encode(text);

                // Ensure input IDs are converted to long (int64)
                var inputIds = ConvertEncodedIds(encoded);

                const int maxSequenceLength = 100; // Maximum number of tokens to generate
                int initialInputSize = inputIds.Count; // Track the initial size of input
                List<long> outputIds = new List<long>();
                GenerateNewIds(inputIds, maxSequenceLength, initialInputSize, outputIds);
                foreach (var id in outputIds)
                {

                    // Decode the token individually
                    string decodedToken = _tokenizer.Decode(new int[] { (int)id });
                    // Append the cleaned token to the result string
                    filteredString.Append(decodedToken);
                }
                var result =  filteredString.ToString();
                var filterAfter = GetStringAfterCC(GetStringAfterCC(result));
                var replacedAfter = ReplaceSpecificChars(filterAfter);
                return replacedAfter;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
                return string.Empty;
            }
        }
        /// <summary>
        /// Replaces characters with specific Unicode values (288 for 'G', 33 for '!') with a space.
        /// </summary>
        private string ReplaceSpecificChars(string input)
        {
            StringBuilder result = new StringBuilder();

            foreach (char c in input)
            {
                int charCode = (int)c;

                // Check if the character is 'G' (Unicode 288) or '!' (Unicode 33)
                if (charCode == 288 || charCode == 33 || charCode == 266)
                {
                    result.Append(' ');  // Replace with a space
                }
                else
                {
                    result.Append(c);  // Keep the original character
                }
            }

            return result.ToString();
        }

        /// <summary>
        /// Returns the substring after two consecutive 'C' characters (Unicode 266).
        /// If 'CC' is not found, returns the original string.
        /// </summary>
        private string GetStringAfterCC(string input)
        {
            // Convert Unicode 266 'C' to char
            char specialC = (char)266;
            string ccPattern = new string(new char[] { specialC, specialC });

            // Find the index of 'CC' pattern in the input string
            int ccIndex = input.IndexOf(ccPattern);

            if (ccIndex != -1)
            {
                // Return the substring after 'CC'
                return input.Substring(ccIndex + 2);
            }

            // If 'CC' is not found, return the original string
            return input;
        }

        private void GenerateNewIds(List<long> inputIds, int maxSequenceLength, int initialInputSize, List<long> outputIds)
        {
            for (int i = 0; i < maxSequenceLength; i++)
            {
                // Create input tensor from current input sequence (long/int64)
                var inputIdsOrtValue = OrtValue.CreateTensorValueFromMemory(inputIds.ToArray(), new long[] { 1, inputIds.Count });

                // Run inference to predict the next token
                var inputs = new Dictionary<string, OrtValue> { { "input_ids", inputIdsOrtValue } };
                var output = inferenceSession.Run(new Microsoft.ML.OnnxRuntime.RunOptions(), inputs, inferenceSession.OutputNames);

                // Process output tensor to get predicted logits
                var tokenActivationOutput = output[0].GetTensorDataAsSpan<float>().ToArray();
                var logits = tokenActivationOutput.Skip((inputIds.Count - 1) * VOCAB_SIZE).Take(VOCAB_SIZE);

                // Apply softmax to get probabilities for the next token
                float sum = logits.Sum(x => (float)Math.Exp(x));
                var softmax = logits.Select(x => (float)Math.Exp(x) / sum).ToArray();

                // Get the predicted token by finding the index with the highest probability
                int predictedToken = Array.IndexOf(softmax, softmax.Max());

                // Break if the predicted token is an end token (e.g., token ID 198)
                if (predictedToken == 198 && i > initialInputSize)
                {
                    break;
                }

                // Append predicted token to input sequence for the next iteration
                inputIds.Add(predictedToken);
                outputIds.Add(predictedToken);
                /*
                // Decode the token individually
                string decodedToken = _tokenizer.Decode(new int[] { predictedToken });

                // Clean the decoded token only for generated tokens (i.e., tokens beyond the initial input)
                if (i >= initialInputSize)
                {
                    // Remove unwanted characters like 'G' and '!'
                    decodedToken = decodedToken.Replace("!", "").Replace("G", "");
                }
                else
                {

                }
                // Append the cleaned token to the result string
                filteredString.Append(decodedToken);
                */
            }
        }

        private static List<long> ConvertEncodedIds(TokenizerResult encoded)
        {
            return encoded.Ids.Select(id => (long)id).ToList();
        }

        /// <summary>
        /// Represents an instance of the ONNX inference session.
        /// </summary>
        internal Microsoft.ML.OnnxRuntime.InferenceSession inferenceSession = null;

        /// <summary>
        /// Represents a task for tokenization.
        /// </summary>
        internal Microsoft.ML.Tokenizers.Tokenizer _tokenizer = null;

        /// <summary>
        /// The size of the vocabulary.
        /// </summary>
        public const int VOCAB_SIZE = 50257;
    }


}