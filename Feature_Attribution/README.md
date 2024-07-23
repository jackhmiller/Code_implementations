## Using Feature Attribution to Change Classification Scores

The setup includes a trained model based on the "MalConv" architecture [1] which is a convolution-based binary classifier with an embedding layer where the inputs are the raw bytes from the files (i.e. *categorical inputs*) and the output is a score between 0 and 1 (0 is benign and 1 is malicious).

We have 5 true positive samples (inactive malicious files with high scores) in the form of a numpy array where each cell represents a byte between 0 and 255 (a special padding character of value 256 is appened at the end if needed but doesn't appear in the original samples you are given). The array is of dimensions 5 (# samples) x 10,000 (trimmed / padded sample size).

We want to modify an input sample to bypass the model (i.e. lower the output score under 0.5) while keeping the modification cost to a minimum (total number of bytes that were modified).

[1] Raff, Edward, et al. "Malware detection by eating a whole exe." arXiv preprint arXiv:1710.09435 (2017).
