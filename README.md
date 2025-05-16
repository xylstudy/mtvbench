## 🛠 How to Use the Prompt-to-Video Mapping Code

### ✅ Step 1: Confirm Prompt Text Format

Ensure your prompt file (e.g., `prompts.txt`) follows this structure:

- Each group contains **2–4 prompts**, each on its own line  
- Each group is separated by a line with `###`  
- No blank lines between prompts or groups

Example:

```
A dog chases a ball in the park, wide shot with green grass surrounding.
A dog sits patiently in the park, wide shot with green grass surrounding.
###
A cat climbs a tree in the yard, medium shot with sunlight filtering through the leaves.
A cat lays on the windowsill, medium shot with sunlight filtering through the leaves.
###
...
```

> ⚠️ **Note:** The order of prompt groups determines the expected video index (e.g., `V_000.mp4`, `V_001.mp4`, ...).  
> Ensure that the **order of videos matches the order of prompt groups** exactly.

---

### ✅ Step 2: Correctly Set Path Variables

Before running the script (e.g., `generate_prompt_mapping.py`), make sure the following paths are set correctly:

```python
PROMPT_PATH = "/path/to/prompts.txt"                      # Path to your prompt text file
VIDEO_DIR = "/path/to/videos"                             # Directory containing the generated videos
OUTPUT_JSON = "/path/to/output/change-alignment_check.json"  # Output file for saving the mapping or score
```

---

### ✅ Step 3: Run the Script

After confirming the format and setting all paths correctly, run the script to generate the mapping file:

```bash
python generate_prompt_mapping.py
```
