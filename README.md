**MTVBench: Benchmarking Video Generation Models with Multiple Transition Text Prompts**

Submission to NeurIPS 2025 Dataset and Benchmark track.

Xiaodong Cun, Xiuli Bi, Ruihuan Yang, Jianfei Yuan, Bin Xiao, Bo Liu.

GVC Lab, Great Bay University, and Chongqing University of Posts and Telecommunications

We list all the generated videos here in each folder.

The prompt benchmark we collected is shown in `mtv_bench_prompt.txt`.


## 🛠 How to Use the Prompt-to-Video Mapping Code

### ✅ Step 1: Confirm Prompt Text Format

Ensure your prompt file (e.g., `mtv_bench_prompt.txt`) follows this structure:

- Each group contains **2–4 prompts**, each on its own line  
- Each group is separated by a line with `###`  
- No blank lines between prompts or groups

Example:

    A dog chases a ball in the park, wide shot with green grass surrounding.
    A dog sits patiently in the park, wide shot with green grass surrounding.
    ###
    A cat climbs a tree in the yard, medium shot with sunlight filtering through the leaves.
    A cat lays on the windowsill, medium shot with sunlight filtering through the leaves.
    ###
    A horse gallops across the field, wide shot with a mountain in the background.
    A horse rests under a tree, wide shot with a mountain in the background.

> This format ensures correct parsing when generating the prompt-to-video JSON mapping.

### ✅ Step 2: Correctly Set Path Variables

Before running the evaluation code (e.g., `semanticalignment_overallalignment.py` )make sure the following paths are correctly set:

```python
PROMPT_PATH = "/path/to/prompts.txt"                      # Path to your prompt text file
VIDEO_DIR = "/path/to/videos"                             # Directory containing the generated videos
OUTPUT_JSON = "/path/to/output/change-alignment_check.json"  # Output file for saving the mapping or score
