---
sidebar_position: 1
---

# Creating a Python Leaderboard
This section describes how to create Python-based leaderboards, which expect Python submissions
(they can still inline compile CUDA code though). To create leaderboards on a Discord server, the
Discord bot expects you to have a `Leaderboard Admin` or `Leaderboard Creator` role. These can be
assigned by admins / owners of the server. Nevertheless, this section is also useful for submitters
to understand the details of a leaderboard under the hood.

Like we've mentioned before, each leaderboard specifies a number of GPUs to evaluate on based on the
creator's choosing. You can think of each `(leaderboard, GPU)` pair as *essentially an independent
leaderboard*, as for example, a softmax kernel on an NVIDIA T4 may be very different on an H100. We
give leaderboard creators the option to select which GPUs they care about for their leaderboards --
for example, they may only care about NVIDIA A100 and NVIDIA H100 performance for their leaderboard.

To create a Python leaderboard given the correct role permissions, you can run (type it out so it fills in
correctly)
<center>
```
/leaderboard create {leaderboard_name: str} {deadline: str} {reference_code: .py file}
```
</center>

After running this, similar to leaderboard submissions, a UI window will pop up asking which GPUs
the leaderboard creator wants to allow submissions on. After selecting the GPUs, the leaderboard
will be created, and users can submit. In the rest of this page, we will explain how to write a
proper `reference_code.py` file to create your own leaderboards.

## The Evaluation Harness
When a user submits a Python kernel submission to your leaderboard, we use the reference code from
the leaderboard and an evalulation script to check for correctness of the user kernel and measure
the runtime. In all:
* `eval.py`: 
* `reference_code.py`: 
* `submission.py`: 
The evaluation harness is the same for all Python leaderboards, and can be retrieved with
<center>
```
/leaderboard eval-code language:python
```
</center>


## Reference Code Requirements
The reference code file **must be a `.py`** to create a Python leaderboard. `.cu, .cuh, .cpp`
reference files willl create [CUDA leaderboards](./cuda-creations). Based on the evaluation harness
above, each reference file **must** have the following function signatures filled out:


```python title="reference_template.py"

def check_implementation(
        user_output: OutputType,
        reference_output: OutputType,
    ) -> bool:
    ...

def generate_input() -> InputType:
    ...

def ref_kernel(data: InputType) -> OutputType:
    ...
```
