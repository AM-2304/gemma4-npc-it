#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
The Goblin Merchant's Bazaar — Gemma4NPC Demo
A Pygame interface where the player negotiates with an AI merchant.
Demonstrates how the LLM's structured JSON output directly controls the game's UI and state.

Usage:
    python demos/goblin_merchant/game.py --model-path models/Gemma4NPC-it-Q4_K_M.gguf
"""

import argparse
import sys
import threading
from pathlib import Path

import pygame

# Make sure we can import the backend
sys.path.append(str(Path(__file__).parent))
from npc_backend import NPCBackend

# === Constants ===
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
FPS = 60

# === Colors ===
BACKGROUND = (20, 20, 25)
PANEL_BG = (35, 35, 45)
TEXT_COLOR = (220, 220, 220)
GOLD_COLOR = (255, 215, 0)
GREEN = (80, 255, 80)
RED = (255, 80, 80)
BLUE = (100, 150, 255)
INPUT_BG = (50, 50, 65)

# === Mood Colors ===
MOOD_COLORS = {
    "greedy": (255, 215, 0),
    "angry": (255, 80, 80),
    "terrified": (150, 150, 255),
    "amused": (255, 150, 255),
    "insulted": (200, 100, 100),
    "intrigued": (100, 255, 200),
    "confused": (150, 150, 150),
}


class Game:
    def __init__(self, model_path: str):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("The Goblin Merchant's Bazaar")
        self.clock = pygame.time.Clock()

        self.font_large = pygame.font.SysFont("Georgia", 36, bold=True)
        self.font_medium = pygame.font.SysFont("Georgia", 24)
        self.font_small = pygame.font.SysFont("Arial", 18)

        # Game State
        self.player_gold = 50
        self.player_inventory = ["A Shiny Rock", "An Old Boot", "Suspicious Purple Potion"]
        
        self.current_price = 500
        self.merchant_mood = "greedy"
        self.merchant_dialogue = "Welcome to Gribble's glorious emporium! The Glowing Amulet is 500 gold. No lowballers!"
        self.game_over = False
        self.game_won = False

        # Input Box
        self.input_text = ""
        self.input_active = True
        self.is_waiting = False  # True when waiting for LLM

        # Chat Log
        self.chat_log = []

        # Load AI Backend
        self.backend = None
        self.loading_thread = threading.Thread(target=self._load_model, args=(model_path,))
        self.loading_thread.start()

    def _load_model(self, model_path):
        self.merchant_dialogue = "Gribble is waking up... (Loading Model, this may take a moment)"
        self.backend = NPCBackend(model_path)
        self.merchant_dialogue = "Welcome to Gribble's glorious emporium! The Glowing Amulet is 500 gold. No lowballers!"

    def _get_npc_response(self, player_message):
        self.is_waiting = True
        try:
            # Add inventory context implicitly so the model knows what the player has
            inventory_context = f"[OOC: The player currently has {self.player_gold} gold and items: {', '.join(self.player_inventory)}]"
            response_data = self.backend.get_response(player_message + "\n" + inventory_context)
            
            self.merchant_dialogue = response_data.get("dialogue", "...")
            self.current_price = response_data.get("current_price", self.current_price)
            self.merchant_mood = response_data.get("merchant_mood", "confused").lower()
            
            self.chat_log.append({"role": "Gribble", "content": self.merchant_dialogue})
            
            if response_data.get("deal_accepted", False):
                self.game_over = True
                self.game_won = True
                
        except Exception as e:
            self.merchant_dialogue = f"*Gribble crashes and burns* Error: {e}"
        finally:
            self.is_waiting = False

    def wrap_text(self, text, font, max_width):
        """Simple text wrapper."""
        words = text.split(' ')
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            width, _ = font.size(' '.join(current_line))
            if width > max_width:
                current_line.pop()
                lines.append(' '.join(current_line))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        return lines

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and not self.game_over:
                    if event.key == pygame.K_RETURN:
                        if self.input_text.strip() and not self.is_waiting and self.backend is not None:
                            msg = self.input_text.strip()
                            self.chat_log.append({"role": "You", "content": msg})
                            self.input_text = ""
                            threading.Thread(target=self._get_npc_response, args=(msg,)).start()
                    elif event.key == pygame.K_BACKSPACE:
                        self.input_text = self.input_text[:-1]
                    else:
                        # Add character if it's printable
                        if event.unicode.isprintable():
                            self.input_text += event.unicode

            self.draw()
            self.clock.tick(FPS)
            
        pygame.quit()

    def draw(self):
        self.screen.fill(BACKGROUND)

        # --- Top UI (Stats) ---
        pygame.draw.rect(self.screen, PANEL_BG, (20, 20, 300, 150), border_radius=10)
        title = self.font_medium.render("Your Inventory", True, TEXT_COLOR)
        self.screen.blit(title, (35, 35))
        
        gold_txt = self.font_small.render(f"Gold: {self.player_gold} coins", True, GOLD_COLOR)
        self.screen.blit(gold_txt, (35, 70))
        
        for i, item in enumerate(self.player_inventory):
            item_txt = self.font_small.render(f"- {item}", True, TEXT_COLOR)
            self.screen.blit(item_txt, (35, 100 + (i * 20)))

        # --- Merchant UI ---
        pygame.draw.rect(self.screen, PANEL_BG, (340, 20, 640, 150), border_radius=10)
        
        merchant_name = self.font_large.render("Gribble the Goblin", True, MOOD_COLORS.get(self.merchant_mood, TEXT_COLOR))
        self.screen.blit(merchant_name, (360, 35))
        
        mood_txt = self.font_small.render(f"Mood: {self.merchant_mood.upper()}", True, MOOD_COLORS.get(self.merchant_mood, TEXT_COLOR))
        self.screen.blit(mood_txt, (360, 80))
        
        price_txt = self.font_large.render(f"Asking Price: {self.current_price} Gold", True, GOLD_COLOR)
        self.screen.blit(price_txt, (650, 50))

        # --- Dialogue Box ---
        pygame.draw.rect(self.screen, PANEL_BG, (20, 190, 960, 150), border_radius=10)
        dialogue_lines = self.wrap_text(f"\" {self.merchant_dialogue} \"", self.font_medium, 900)
        for i, line in enumerate(dialogue_lines):
            line_surface = self.font_medium.render(line, True, TEXT_COLOR)
            self.screen.blit(line_surface, (50, 210 + (i * 35)))

        # --- Chat Log ---
        pygame.draw.rect(self.screen, PANEL_BG, (20, 360, 960, 240), border_radius=10)
        y_offset = 570
        for entry in reversed(self.chat_log[-6:]):  # Show last 6 messages
            color = BLUE if entry["role"] == "You" else GREEN
            role_txt = self.font_small.render(f"{entry['role']}: ", True, color)
            msg_txt = self.font_small.render(entry["content"], True, TEXT_COLOR)
            self.screen.blit(role_txt, (35, y_offset))
            self.screen.blit(msg_txt, (35 + role_txt.get_width(), y_offset))
            y_offset -= 30

        # --- Input Box ---
        if self.game_over:
            pygame.draw.rect(self.screen, INPUT_BG, (20, 620, 960, 60), border_radius=10)
            msg = "DEAL ACCEPTED! You acquired the amulet!" if self.game_won else "GAME OVER"
            color = GREEN if self.game_won else RED
            txt = self.font_medium.render(msg, True, color)
            self.screen.blit(txt, (SCREEN_WIDTH//2 - txt.get_width()//2, 635))
        else:
            pygame.draw.rect(self.screen, INPUT_BG, (20, 620, 960, 60), border_radius=10)
            pygame.draw.rect(self.screen, BLUE, (20, 620, 960, 60), 2, border_radius=10)
            
            display_text = self.input_text
            if self.is_waiting:
                display_text = "Gribble is thinking..."
                
            input_surface = self.font_medium.render(display_text, True, TEXT_COLOR if not self.is_waiting else GRAY)
            self.screen.blit(input_surface, (35, 635))
            
            # Cursor
            if not self.is_waiting and (pygame.time.get_ticks() % 1000 < 500):
                cursor_x = 35 + input_surface.get_width()
                pygame.draw.line(self.screen, TEXT_COLOR, (cursor_x, 635), (cursor_x, 660), 2)

        pygame.display.flip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to GGUF model")
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"Error: Model not found at {args.model_path}")
        exit(1)

    game = Game(args.model_path)
    game.run()

if __name__ == "__main__":
    main()
