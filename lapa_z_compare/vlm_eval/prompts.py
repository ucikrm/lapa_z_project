"""Prompt templates for VLM-based task-progress scoring."""

# TOPReward-style completion query. We read the probability of the affirmative
# token ("True") for this binary judgment, following Chen et al. (TOPReward).
# Adapted for STHV2 human-manipulation clips (single short action per clip).
TOPREWARD_QUERY = (
    "The frames above are from a video of a person performing the task: "
    "\"{instruction}\". Compared to the first frame, does the last frame show this "
    "task being completed or making clear visual progress? "
    "Answer with a single word, True or False."
)

# GVL-style autoregressive progress prompt (used for multi-frame scorer validation).
GVL_SYSTEM = (
    "You are an expert at estimating task progress from video frames for the task: "
    "\"{instruction}\". For each frame, output an integer task-completion percentage "
    "between 0 and 100, where 0 is the start and 100 is full completion. "
    "The frames are shown in RANDOM order, so judge each frame on its own visual content."
)
GVL_FRAME_REQUEST = (
    "Now output the completion percentage for each of the following shuffled frames. "
    "Respond with one line per frame formatted exactly as 'Frame {i}: P%'."
)
