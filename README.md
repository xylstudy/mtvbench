**MTVBench: Benchmarking Video Generation Models with Multiple Transition Text Prompts**

Submission to NeurIPS 2025 Dataset and Benchmark track.

Xiaodong Cun, Xiuli Bi, Ruihuan Yang, Jianfei Yuan, Bin Xiao, Bo Liu.

GVC Lab, Great Bay University, and Chongqing University of Posts and Telecommunications

We list all the generated videos here in each folder.

The prompt benchmark we collected is shown in `mtv_bench_prompt.txt`.


## 🛠 How to Use the Prompt-to-Video Mapping Code

### ✅ Step 1: Confirm Prompt Text Format

Ensure your prompt file (e.g., `mtv_bench_prompt.txt`) follows the correct structure:

- Each group of **2–4 prompts** is separated by `###`
- Each prompt is a **single line**
- **No empty lines** between prompts

Example:

A dog chases a ball in the park, wide shot with green grass surrounding.
A dog sits patiently in the park, wide shot with green grass surrounding.
###
A cat climbs a tree in the yard, medium shot with sunlight filtering through the leaves.
A cat lays on the windowsill, medium shot with sunlight filtering through the leaves.


> This format is required for downstream mapping and evaluation tools to parse the prompts correctly.

