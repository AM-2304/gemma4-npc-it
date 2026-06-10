import json
import os

def fix_alternating_roles(input_path, output_path):
    valid_data = []
    dropped_count = 0
    
    print(f"Reading from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            messages = item.get("messages", [])
            
            if not messages:
                dropped_count += 1
                continue
                
            # Filter out non-alternating messages by squashing consecutive roles
            # or just dropping invalid conversations. Squashing is better.
            
            fixed_messages = []
            for msg in messages:
                role = msg["role"]
                # Map 'assistant' to 'model' if necessary, though 'assistant' is standard
                
                if not fixed_messages:
                    fixed_messages.append(msg)
                else:
                    prev_role = fixed_messages[-1]["role"]
                    
                    # If same role, append text
                    if role == prev_role:
                        fixed_messages[-1]["content"] += "\n" + msg["content"]
                    else:
                        # Ensure strict alternation (System -> User -> Assistant -> User -> Assistant)
                        # The system prompt is fine at the start.
                        if prev_role == "system" and role != "user":
                            # We expect user after system. If assistant comes after system, we drop or fix.
                            # Just drop it to be safe for DPO/SFT
                            fixed_messages = []
                            break
                        
                        if prev_role == "user" and role != "assistant" and role != "model":
                            fixed_messages = []
                            break
                            
                        if (prev_role == "assistant" or prev_role == "model") and role != "user":
                            fixed_messages = []
                            break
                            
                        fixed_messages.append(msg)
            
            # Must end with assistant response for training
            if fixed_messages and fixed_messages[-1]["role"] in ["assistant", "model"]:
                valid_data.append({"messages": fixed_messages})
            else:
                dropped_count += 1
                
    with open(output_path, "w", encoding="utf-8") as f:
        for item in valid_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Saved {len(valid_data)} strictly alternating conversations to {output_path}")
    print(f"Dropped or squashed {dropped_count} invalid conversations.")

if __name__ == "__main__":
    input_file = "data/augmented/npc_quest_sft.jsonl"
    output_file = "data/augmented/npc_quest_sft_fixed.jsonl"
    fix_alternating_roles(input_file, output_file)
