def minibatch_weight(batch_idx : int, num_batches : int):
    return 2 ** (num_batches - batch_idx) / (2 **num_batches - batch_idx)