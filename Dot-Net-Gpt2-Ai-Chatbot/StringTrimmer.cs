using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nexez
{
    internal class StringTrimmer
    {
        // Method to remove "Ai Assistant:" or "User:" from the beginning of a string
        internal static string TrimPrefix(string input)
        {
            string[] prefixes = { "Ai Assistant:", "User:" };
            foreach (string prefix in prefixes)
            {
                if (input.StartsWith(prefix))
                {
                    input = input.Substring(prefix.Length);
                    break;
                }
            }
            string[] prefixes2 = { "Ai Assistant", "User" };
            foreach (string prefix in prefixes2)
            {
                if (input.StartsWith(prefix))
                {
                    input = input.Substring(prefix.Length);
                    break;
                }
            }

            return input.TrimStart(); // Trim any leading whitespace
        }

        // Method to remove "Ai Assistant:" or "User:" from the end of a string
        internal static string TrimSuffix(string input)
        {
            string[] suffixes = { "Ai Assistant:", "User:" };
            string[] suffixes2 = { "Ai Assistant", "User" };
            foreach (string suffix in suffixes)
            {
                if (input.EndsWith(suffix))
                {
                    input = input.Substring(0, input.Length - suffix.Length);
                    break;
                }
            }
            foreach (string suffix in suffixes2)
            {
                if (input.EndsWith(suffix))
                {
                    input = input.Substring(0, input.Length - suffix.Length);
                    break;
                }
            }
            return input.TrimEnd(); // Trim any trailing whitespace
        }
    }
}