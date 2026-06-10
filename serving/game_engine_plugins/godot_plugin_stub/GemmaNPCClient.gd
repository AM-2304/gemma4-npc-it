# GemmaNPCClient.gd
# Godot 4 plugin for Gemma4NPC integration
# SPDX-License-Identifier: Apache-2.0
# Drop into your project's addons folder and configure NPC_SERVER_URL
extends Node

const NPC_SERVER_URL = "http://localhost:8080/v1/chat/completions"
var http_request: HTTPRequest
var conversation_history: Array = []
var system_prompt: String = ""
var max_history_turns: int = 10

signal npc_responded(response: String)
signal npc_error(error: String)

func _ready() -> void:
	http_request = HTTPRequest.new()
	add_child(http_request)
	http_request.request_completed.connect(_on_request_completed)

func initialize_npc(npc_system_prompt: String) -> void:
	"""Set up the NPC with a character definition."""
	system_prompt = npc_system_prompt
	conversation_history.clear()

func get_npc_response(player_message: String) -> void:
	"""Send player message and get NPC response asynchronously."""
	var messages = [{"role": "system", "content": system_prompt}]

	# Add sliding window history
	var start_idx = max(0, conversation_history.size() - max_history_turns)
	for i in range(start_idx, conversation_history.size()):
		messages.append(conversation_history[i])

	messages.append({"role": "user", "content": player_message})
	conversation_history.append({"role": "user", "content": player_message})

	var body = JSON.stringify({
		"model": "gemma4npc-it",
		"messages": messages,
		"temperature": 1.0,
		"max_tokens": 150
	})

	var error = http_request.request(
		NPC_SERVER_URL,
		["Content-Type: application/json"],
		HTTPClient.METHOD_POST,
		body
	)

	if error != OK:
		npc_error.emit("Failed to send request: " + str(error))

func _on_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray) -> void:
	if response_code != 200:
		npc_error.emit("Server error: " + str(response_code))
		return

	var json = JSON.new()
	var parse_result = json.parse(body.get_string_from_utf8())
	if parse_result != OK:
		npc_error.emit("Failed to parse response")
		return

	var data = json.data
	var response_text = data["choices"][0]["message"]["content"]

	conversation_history.append({"role": "assistant", "content": response_text})
	npc_responded.emit(response_text)

func clear_history() -> void:
	"""Reset conversation history while keeping the system prompt."""
	conversation_history.clear()
