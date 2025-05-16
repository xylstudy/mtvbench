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

> ⚠️ **Note:** The order of prompt groups determines the expected video index (e.g., `V_1.mp4`, `V_2.mp4`, ...).  
> Ensure that the **order of videos matches the order of prompt groups** exactly.

---

### ✅ Step 2: Correctly Set Path Variables

Before running the script (e.g., `generate_prompt_mapping.py`), make sure the following paths are set correctly:

```python
PROMPT_PATH = "/path/to/prompts.txt"                         # Path to your prompt text file
VIDEO_DIR = "/path/to/videos"                                # Directory containing the generated videos
OUTPUT_JSON = "/path/to/output/change-alignment_check.json"  # Output file for saving the mapping or score
BASE_MODEL_PATH  = "/path/to/model"                          # Path to Qwen-VL or CLIP model checkpoint
PROMPT_META_PATH    = "/path/to/prompts_meta.json"       # Metadata JSON file with action1/action2 per group
```

---

### ✅ Step 3: Run the Script

After confirming the format and setting all paths correctly, run the script to generate the mapping file:

```bash
python generate_prompt_mapping.py
```
---

### 🧾 Prompt Metadata Format (`prompts_meta.json`)

This file provides **structured information about action transitions** for each prompt group. It is required for evaluating **attribute-level alignment** (e.g., action consistency).

#### 📘 JSON Structure

Each key corresponds to a prompt group index (starting from 1), and maps to the actions associated with that group:

```json
{
    "1": {
        "action1": "chases",
        "action2": "sits patiently"
    },
    "2": {
        "action1": "climbs",
        "action2": "lays"
    }
}
```

#### 📌 Notes

- The keys `"action1"` and `"action2"` refer to the **first** and **last** actions in the multi-prompt sequence, respectively.
- Only two actions are currently supported; for longer sequences, the script may need to be extended accordingly.
- This file must be consistent with the order and grouping defined in `prompts.txt`.

---
