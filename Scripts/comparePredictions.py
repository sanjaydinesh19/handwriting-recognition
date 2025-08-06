from mltu.utils.text_utils import get_cer, get_wer

# Load model prediction
with open("test_predictions.txt", "r", encoding="utf-8") as f:
    prediction = f.read().strip()

# Original ground truth
ground_truth = (
    "The lovely Spring night may song luna shines Welcome and farewell my heart was beating. "
    "The rosebush on the moor the violet beautiful The artist's evening song new love new life "
    "To Belinda Holda this roast so long farewell Now I have this little hut where my beloved live "
    "Walking now with veiled steps through the leaves Luna shines through lush and oak zephyr penetrate "
    "And the birch trees bowing low shed incense on the track How beautiful the coolness of this lovely summer night! "
    "How the soul fills with happiness in this true place of quiet! I can scarcely grasp the bliss, yet Heaven, I would shun "
    "A thousand nights like this if my darling granted one."
)
words = ground_truth.split()
print("Word count:", len(words))

# Calculate metrics
cer = get_cer([prediction], [ground_truth])
wer = get_wer([prediction], [ground_truth])


# Print results
print(f"Character Error Rate (CER): {cer:.4f}")
print(f"Word Error Rate (WER): {wer:.4f}")
with open("errors.txt", "w", encoding="utf-8") as f:
    for pred, true in zip(prediction.split(), ground_truth.split()):
        if pred != true:
            f.write(f"❌ Predicted: {pred} | ✅ Actual: {true}\n")

