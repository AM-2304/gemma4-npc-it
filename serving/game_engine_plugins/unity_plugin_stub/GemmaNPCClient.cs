// GemmaNPCClient.cs — Production-Ready Unity Plugin for Gemma4NPC
// SPDX-License-Identifier: Apache-2.0
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.Events;

namespace Gemma4NPC.Unity
{
    /// <summary>
    /// Production-ready Unity client for the Gemma4NPC OpenAI-compatible API.
    /// Manages conversation history, handles network timeouts, and provides UnityEvents
    /// for easy integration with UI components (like the Goblin Bazaar).
    /// </summary>
    public class GemmaNPCClient : MonoBehaviour
    {
        [Header("Network Configuration")]
        [Tooltip("The endpoint of your local or cloud OpenAI-compatible server.")]
        [SerializeField] private string serverUrl = "http://localhost:8080/v1/chat/completions";
        [Tooltip("Timeout in seconds for the inference request.")]
        [SerializeField] private int requestTimeout = 30;

        [Header("NPC Identity Context")]
        [TextArea(5, 15)]
        [SerializeField] private string systemPrompt = "You are Gringo, a cunning but friendly Goblin Merchant in a fantasy bazaar. You are trying to sell magical artifacts to the player. Keep your responses short, quirky, and in-character.";
        [SerializeField] private int maxHistoryTurns = 12;

        [Header("Generation Settings")]
        [Range(0.1f, 2.0f)]
        [SerializeField] private float temperature = 0.8f;
        [SerializeField] private int maxTokens = 200;

        [Header("Events")]
        public UnityEvent OnThinkingStarted;
        public UnityEvent<string> OnResponseReceived;
        public UnityEvent<string> OnError;

        private List<ChatMessage> conversationHistory = new List<ChatMessage>();
        private bool isGenerating = false;

        public bool IsGenerating => isGenerating;

        private void Start()
        {
            // Initialize history
            ClearHistory();
        }

        /// <summary>
        /// Sends a message from the player to the NPC.
        /// </summary>
        /// <param name="playerMessage">The text input from the player.</param>
        public void SendPlayerMessage(string playerMessage)
        {
            if (isGenerating)
            {
                Debug.LogWarning("[GemmaNPC] Already waiting for a response. Ignoring input.");
                return;
            }

            if (string.IsNullOrWhiteSpace(playerMessage)) return;

            StartCoroutine(GenerateNPCResponseRoutine(playerMessage));
        }

        private IEnumerator GenerateNPCResponseRoutine(string playerMessage)
        {
            isGenerating = true;
            OnThinkingStarted?.Invoke();

            // 1. Build the payload
            var messages = new List<ChatMessage>
            {
                new ChatMessage { role = "system", content = systemPrompt }
            };

            // Append sliding window history
            int startIdx = Mathf.Max(0, conversationHistory.Count - maxHistoryTurns);
            for (int i = startIdx; i < conversationHistory.Count; i++)
            {
                messages.Add(conversationHistory[i]);
            }

            // Append current user message
            messages.Add(new ChatMessage { role = "user", content = playerMessage });

            var requestBody = JsonUtility.ToJson(new ChatCompletionRequest
            {
                model = "gemma4npc-it",
                messages = messages.ToArray(),
                temperature = temperature,
                max_tokens = maxTokens
            });

            // 2. Send the Request
            using (var request = new UnityWebRequest(serverUrl, "POST"))
            {
                byte[] bodyRaw = Encoding.UTF8.GetBytes(requestBody);
                request.uploadHandler = new UploadHandlerRaw(bodyRaw);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");
                request.timeout = requestTimeout;

                yield return request.SendWebRequest();

                if (request.result == UnityWebRequest.Result.ConnectionError || request.result == UnityWebRequest.Result.ProtocolError)
                {
                    string errorMsg = $"[GemmaNPC] Network Error: {request.error}";
                    Debug.LogError(errorMsg);
                    OnError?.Invoke(errorMsg);
                    isGenerating = false;
                    yield break;
                }

                // 3. Parse Response
                string responseText = ParseResponse(request.downloadHandler.text);

                if (!string.IsNullOrEmpty(responseText))
                {
                    // Save to history only if successful
                    conversationHistory.Add(new ChatMessage { role = "user", content = playerMessage });
                    conversationHistory.Add(new ChatMessage { role = "assistant", content = responseText });

                    OnResponseReceived?.Invoke(responseText);
                }
                else
                {
                    OnError?.Invoke("Failed to parse NPC response.");
                }
            }

            isGenerating = false;
        }

        private string ParseResponse(string json)
        {
            try
            {
                var response = JsonUtility.FromJson<ChatCompletionResponse>(json);
                if (response != null && response.choices != null && response.choices.Length > 0)
                {
                    return response.choices[0].message.content.Trim();
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[GemmaNPC] JSON Parse Error: {e.Message}\nRaw JSON: {json}");
            }
            return string.Empty;
        }

        /// <summary>
        /// Clears the active conversation memory.
        /// </summary>
        public void ClearHistory()
        {
            conversationHistory.Clear();
            Debug.Log("[GemmaNPC] Conversation history cleared.");
        }

        // --- JSON Serialization Data Structures ---
        
        [Serializable] 
        private class ChatMessage 
        { 
            public string role; 
            public string content; 
        }

        [Serializable] 
        private class ChatCompletionRequest 
        { 
            public string model; 
            public ChatMessage[] messages; 
            public float temperature; 
            public int max_tokens; 
        }

        [Serializable] 
        private class ChatCompletionResponse 
        { 
            public Choice[] choices; 
        }

        [Serializable] 
        private class Choice 
        { 
            public ChatMessage message; 
            public string finish_reason; 
        }
    }
}
