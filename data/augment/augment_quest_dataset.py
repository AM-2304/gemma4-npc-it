import json
from datasets import load_dataset
import os

def augment_npc_dataset():
    print("Loading chimbiwide/RolePlay-NPC-Quest...")
    # Load the community dataset
    ds = load_dataset('chimbiwide/RolePlay-NPC-Quest')
    
    augmented_data = []
    
    print("Augmenting and injecting JSON Schema...")
    for idx, row in enumerate(ds['train']):
        messages = row['messages']
        
        new_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                # Enhance their system prompt with our strict JSON constraints
                enhanced_system = msg['content'] + """

[ENGINE INTEGRATION]
You must ALWAYS respond in valid JSON format.
{
  "dialogue": "Your spoken text",
  "quest_status": "none" // or "assigned", "completed" if applicable
}"""
                new_messages.append({"role": "system", "content": enhanced_system})
                
            elif msg['role'] == 'user':
                new_messages.append(msg)
                
            elif msg['role'] == 'assistant':
                # Wrap their plain-text dialogue into our required JSON schema
                raw_text = msg['content']
                
                # Simple heuristic: if the raw text sounds like a quest, flag it!
                quest_status = "none"
                text_lower = raw_text.lower()
                if "quest" in text_lower or "mission" in text_lower or "find" in text_lower or "bring me" in text_lower:
                    quest_status = "assigned"
                if "thank you" in text_lower and "reward" in text_lower:
                    quest_status = "completed"
                    
                json_content = {
                    "dialogue": raw_text,
                    "quest_status": quest_status
                }
                
                new_messages.append({
                    "role": "assistant", 
                    "content": json.dumps(json_content, indent=2)
                })
        
        augmented_data.append({"messages": new_messages})
            
    # Save to a new JSONL file ready for SFT
    os.makedirs("data/augmented", exist_ok=True)
    output_path = "data/augmented/npc_quest_sft.jsonl"
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in augmented_data:
            f.write(json.dumps(item) + "\n")
            
    print(f"\nSuccessfully augmented {len(augmented_data)} conversations!")
    print(f"Saved to: {output_path}")
    
    # Print a beautiful example of what we just did
    print("\n--- BEFORE (Theirs) ---")
    print(ds['train'][0]['messages'][2]['content'])
    print("\n--- AFTER (Ours) ---")
    print(augmented_data[0]['messages'][2]['content'])

if __name__ == "__main__":
    augment_npc_dataset()
