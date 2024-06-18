import torch

from torch.nn import functional as F


def ctc_loss(loss_fn, logits, targets):
    logits = logits.mean(dim=2)
    # Calculate each sequence length for each sample
    sample_batch_size, sequence_length = logits.size(0), logits.size(1)
    input_lengths = torch.full(size=(sample_batch_size,), fill_value=sequence_length, dtype=torch.long)
    # Calculate target length for each target sample
    target_lengths = targets.ne(0).sum(dim=1)
    # Transpose the logits
    logits = logits.permute(2, 0, 1)
    log_probs = F.log_softmax(logits, dim=-1)
    # Calculate loss
    loss = loss_fn(log_probs=log_probs,
                   targets=targets,
                   input_lengths=input_lengths,
                   target_lengths=target_lengths)
    return loss
