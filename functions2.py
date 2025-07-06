def replace_line_at_index(filename, index, new_line):
    """
    Replaces the line at the given index in the file with new_line.

    Parameters:
        filename (str): Path to the file.
        index (int): Zero-based line index to replace.
        new_line (str): The new line to insert (should end with '\n' if needed).

    Returns:
        str: The filename (unchanged).
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # Ensure the list is long enough
    if index < len(lines):
        lines[index] = new_line
    else:
        # Pad with empty lines if necessary
        while len(lines) < index:
            lines.append("\n")
        lines.append(new_line)

    with open(filename, "w") as f:
        f.writelines(lines)

    return filename
