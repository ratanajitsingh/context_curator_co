import asyncio
from ollama_backend import qwen_compress, qwen_loss_score, OllamaError

SAMPLE_TEXT = (
    "The board approved the merger contingent on three conditions: first, "
    "regulatory approval from both the FTC and the EU competition authority "
    "by March 15th; second, shareholder approval exceeding 67%; and third, "
    "no material adverse change in either company's debt covenants. CFO "
    "Maria Chen noted the deal is expected to close in Q3, pending all "
    "three conditions, with an estimated $2.3B in synergies over five years."
)

async def main():
    print("Testing qwen_compress (4b)...")
    try:
        compressed = await qwen_compress(SAMPLE_TEXT, target_tokens=15)
        print("COMPRESSED OUTPUT:")
        print(compressed)
        print()
    except OllamaError as e:
        print(f"FAILED: {e}")
        return

    print("Testing qwen_loss_score (9b)...")
    try:
        score = await qwen_loss_score(SAMPLE_TEXT, compressed)
        print(f"LOSS SCORE: {score}")
    except OllamaError as e:
        print(f"FAILED: {e}")

asyncio.run(main())