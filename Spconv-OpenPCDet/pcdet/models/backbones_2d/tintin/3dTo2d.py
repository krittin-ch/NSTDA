def compress_and_normalize(data, dim_to_compress=0):
    """
    Compresses one dimension of a 3D tensor to 2D by summing along that dimension
    and then normalizes the resulting 2D representation using min-max normalization.

    Args:
        data (torch.Tensor): Input 3D tensor of shape (depth, height, width).
        dim_to_compress (int): Dimension to compress (default: 0).

    Returns:
        compressed_normalized_data (torch.Tensor): Compressed and normalized 2D tensor.
    """
    # Sum along the specified dimension to compress it
    compressed_data = torch.sum(data, dim=dim_to_compress)

    # Normalize the compressed data using min-max normalization
    min_val = compressed_data.min()
    max_val = compressed_data.max()
    normalized_data = (compressed_data - min_val) / (max_val - min_val)

    return normalized_data