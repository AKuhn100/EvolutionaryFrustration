import numpy as np
import pandas as pd
from Bio import SeqIO
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Define global variables for worker processes
global_mj_matrix = None
global_coupling_matrix = None
global_aa_to_index = None

# Define the MJ coupling score matrix scaffold
amino_acids = ['C', 'M', 'F', 'I', 'L', 'V', 'W', 'Y', 
              'A', 'G', 'T', 'S', 'N', 'Q', 'D', 'E', 
              'H', 'R', 'K', 'P']
aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}
index_to_aa = {idx: aa for aa, idx in aa_to_index.items()}
num_aa = len(amino_acids)

# Initialize the MJ matrix as a NumPy array for faster access
mj_matrix = np.zeros((num_aa, num_aa))

# Fill in the MJ matrix based on the provided data
mj_entries = {
    # C row
    ('C', 'C'): -5.44, ('C', 'M'): -4.99, ('C', 'F'): -5.80, ('C', 'I'): -5.50,
    ('C', 'L'): -5.83, ('C', 'V'): -4.96, ('C', 'W'): -4.95, ('C', 'Y'): -4.16,
    ('C', 'A'): -3.57, ('C', 'G'): -3.16, ('C', 'T'): -3.11, ('C', 'S'): -2.86,
    ('C', 'N'): -2.59, ('C', 'Q'): -2.85, ('C', 'D'): -2.41, ('C', 'E'): -2.27,
    ('C', 'H'): -3.60, ('C', 'R'): -2.57, ('C', 'K'): -1.95, ('C', 'P'): -3.07,
    
    # M row
    ('M', 'M'): -5.46, ('M', 'F'): -6.56, ('M', 'I'): -6.02, ('M', 'L'): -6.41,
    ('M', 'V'): -5.32, ('M', 'W'): -5.55, ('M', 'Y'): -4.91, ('M', 'A'): -3.94,
    ('M', 'G'): -3.39, ('M', 'T'): -3.51, ('M', 'S'): -3.03, ('M', 'N'): -2.95,
    ('M', 'Q'): -3.30, ('M', 'D'): -2.57, ('M', 'E'): -2.89, ('M', 'H'): -3.98,
    ('M', 'R'): -3.12, ('M', 'K'): -2.48, ('M', 'P'): -3.45,
    
    # F row
    ('F', 'F'): -7.26, ('F', 'I'): -6.84, ('F', 'L'): -7.28, ('F', 'V'): -6.29,
    ('F', 'W'): -6.16, ('F', 'Y'): -5.66, ('F', 'A'): -4.81, ('F', 'G'): -4.13,
    ('F', 'T'): -4.28, ('F', 'S'): -4.02, ('F', 'N'): -3.75, ('F', 'Q'): -4.10,
    ('F', 'D'): -3.48, ('F', 'E'): -3.56, ('F', 'H'): -4.77, ('F', 'R'): -3.98,
    ('F', 'K'): -3.36, ('F', 'P'): -4.25,
    
    # I row
    ('I', 'I'): -6.54, ('I', 'L'): -7.04, ('I', 'V'): -6.05, ('I', 'W'): -5.78,
    ('I', 'Y'): -5.25, ('I', 'A'): -4.58, ('I', 'G'): -3.78, ('I', 'T'): -4.03,
    ('I', 'S'): -3.52, ('I', 'N'): -3.24, ('I', 'Q'): -3.67, ('I', 'D'): -3.17,
    ('I', 'E'): -3.27, ('I', 'H'): -4.14, ('I', 'R'): -3.63, ('I', 'K'): -3.01,
    ('I', 'P'): -3.76,
    
    # L row
    ('L', 'L'): -7.37, ('L', 'V'): -6.48, ('L', 'W'): -6.14, ('L', 'Y'): -5.67,
    ('L', 'A'): -4.91, ('L', 'G'): -4.16, ('L', 'T'): -4.34, ('L', 'S'): -3.92,
    ('L', 'N'): -3.74, ('L', 'Q'): -4.04, ('L', 'D'): -3.40, ('L', 'E'): -3.59,
    ('L', 'H'): -4.54, ('L', 'R'): -4.03, ('L', 'K'): -3.37, ('L', 'P'): -4.20,
    
    # V row
    ('V', 'V'): -5.52, ('V', 'W'): -5.18, ('V', 'Y'): -4.62, ('V', 'A'): -4.04,
    ('V', 'G'): -3.38, ('V', 'T'): -3.46, ('V', 'S'): -3.05, ('V', 'N'): -2.83,
    ('V', 'Q'): -3.07, ('V', 'D'): -2.48, ('V', 'E'): -2.67, ('V', 'H'): -3.58,
    ('V', 'R'): -3.07, ('V', 'K'): -2.49, ('V', 'P'): -3.32,
    
    # W row
    ('W', 'W'): -5.06, ('W', 'Y'): -4.66, ('W', 'A'): -3.82, ('W', 'G'): -3.42,
    ('W', 'T'): -3.22, ('W', 'S'): -2.99, ('W', 'N'): -3.07, ('W', 'Q'): -3.11,
    ('W', 'D'): -2.84, ('W', 'E'): -2.99, ('W', 'H'): -3.98, ('W', 'R'): -3.41,
    ('W', 'K'): -2.69, ('W', 'P'): -3.73,
    
    # Y row
    ('Y', 'Y'): -4.17, ('Y', 'A'): -3.36, ('Y', 'G'): -3.01, ('Y', 'T'): -3.01,
    ('Y', 'S'): -2.78, ('Y', 'N'): -2.76, ('Y', 'Q'): -2.97, ('Y', 'D'): -2.76,
    ('Y', 'E'): -2.79, ('Y', 'H'): -3.52, ('Y', 'R'): -3.16, ('Y', 'K'): -2.60,
    ('Y', 'P'): -3.19,
    
    # A row
    ('A', 'A'): -2.72, ('A', 'G'): -2.31, ('A', 'T'): -2.32, ('A', 'S'): -2.01,
    ('A', 'N'): -1.84, ('A', 'Q'): -1.89, ('A', 'D'): -1.70, ('A', 'E'): -1.51,
    ('A', 'H'): -2.41, ('A', 'R'): -1.83, ('A', 'K'): -1.31, ('A', 'P'): -2.03,
    
    # G row
    ('G', 'G'): -2.24, ('G', 'T'): -2.08, ('G', 'S'): -1.82, ('G', 'N'): -1.74,
    ('G', 'Q'): -1.66, ('G', 'D'): -1.59, ('G', 'E'): -1.22, ('G', 'H'): -2.15,
    ('G', 'R'): -1.72, ('G', 'K'): -1.15, ('G', 'P'): -1.87,
    
    # T row
    ('T', 'T'): -2.12, ('T', 'S'): -1.96, ('T', 'N'): -1.88, ('T', 'Q'): -1.90,
    ('T', 'D'): -1.80, ('T', 'E'): -1.74, ('T', 'H'): -2.42, ('T', 'R'): -1.90,
    ('T', 'K'): -1.31, ('T', 'P'): -1.90,
    
    # S row
    ('S', 'S'): -1.67, ('S', 'N'): -1.58, ('S', 'Q'): -1.49, ('S', 'D'): -1.63,
    ('S', 'E'): -1.48, ('S', 'H'): -2.11, ('S', 'R'): -1.62, ('S', 'K'): -1.05,
    ('S', 'P'): -1.57,
    
    # N row
    ('N', 'N'): -1.68, ('N', 'Q'): -1.71, ('N', 'D'): -1.68, ('N', 'E'): -1.51,
    ('N', 'H'): -2.08, ('N', 'R'): -1.64, ('N', 'K'): -1.21, ('N', 'P'): -1.53,
    
    # Q row
    ('Q', 'Q'): -1.54, ('Q', 'D'): -1.46, ('Q', 'E'): -1.42, ('Q', 'H'): -1.98,
    ('Q', 'R'): -1.80, ('Q', 'K'): -1.29, ('Q', 'P'): -1.73,
    
    # D row
    ('D', 'D'): -1.21, ('D', 'E'): -1.02, ('D', 'H'): -2.32, ('D', 'R'): -2.29,
    ('D', 'K'): -1.68, ('D', 'P'): -1.33,
    
    # E row
    ('E', 'E'): -0.91, ('E', 'H'): -2.15, ('E', 'R'): -2.27, ('E', 'K'): -1.80,
    ('E', 'P'): -1.26,
    
    # H row
    ('H', 'H'): -3.05, ('H', 'R'): -2.16, ('H', 'K'): -1.35, ('H', 'P'): -2.25,
    
    # R row
    ('R', 'R'): -1.55, ('R', 'K'): -0.59, ('R', 'P'): -1.70,
    
    # K row
    ('K', 'K'): -0.12, ('K', 'P'): -0.97,
    
    # P row
    ('P', 'P'): -1.75
}

# Populate the MJ matrix ensuring symmetry
for (aa1, aa2), value in mj_entries.items():
    if aa1 not in aa_to_index or aa2 not in aa_to_index:
        raise ValueError(f"Invalid amino acid pair: ({aa1}, {aa2})")
    idx1, idx2 = aa_to_index[aa1], aa_to_index[aa2]
    mj_matrix[idx1, idx2] = value
    mj_matrix[idx2, idx1] = value  # Ensure symmetry

def load_fasta(filename):
    """
    Load the protein sequence from a FASTA file.
    """
    with open(filename, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            return str(record.seq)
    raise ValueError("No records found in the FASTA file.")

def load_coupling_scores(filename, num_residues):
    """
    Load and process the coupling scores from a CSV file.
    Returns a coupling score matrix.
    """
    df = pd.read_csv(filename)
    
    # Normalize 'cn' between 0 and 1
    cn_min = df['cn'].min()
    cn_max = df['cn'].max()
    df['cn_normalized'] = (df['cn'] - cn_min) / (cn_max - cn_min)
    
    # Multiply 'cn_normalized' by 'probability'
    df['weighted_cn'] = df['cn_normalized'] * df['probability']
    
    # Initialize a coupling score matrix with 1-based indexing
    coupling_matrix = np.zeros((num_residues + 1, num_residues + 1))
    
    for _, row in df.iterrows():
        i, j, weighted_cn = int(row['i']), int(row['j']), row['weighted_cn']
        coupling_matrix[i, j] = weighted_cn
        coupling_matrix[j, i] = weighted_cn  # Ensure symmetry
    
    return coupling_matrix

def calculate_mj_score(sequence_indices, mj_matrix, coupling_matrix):
    """
    Calculate the MJ score for a given sequence.
    """
    score = 0.0
    num_residues = len(sequence_indices)
    for i in range(num_residues):
        for j in range(i + 1, num_residues):
            coupling_score = coupling_matrix[i + 1, j + 1]  # 1-based indexing
            if coupling_score != 0:
                aa1, aa2 = sequence_indices[i], sequence_indices[j]
                score += mj_matrix[aa1, aa2] * coupling_score
    return score

def generate_mutations(sequence_list, amino_acids):
    """
    Generate all possible single amino acid mutations.
    Returns a list of tuples: (wt_residue, position, mutant_residue, mutated_seq)
    """
    mutations = []
    for i, original_aa in enumerate(sequence_list):
        for aa in amino_acids:
            if aa != original_aa:
                mutated_seq = sequence_list.copy()
                mutated_seq[i] = aa
                mutations.append((original_aa, i + 1, aa, mutated_seq))
    return mutations

def calculate_single_mutation_score(args):
    """
    Calculate the MJ score for a single mutation.
    Args is a tuple containing:
    (original_sequence_indices, original_score, mutation, mj_matrix, coupling_matrix, aa_to_index)
    Returns:
    (mutation_label, mutated_score)
    """
    original_sequence_indices, original_score, mutation, mj_matrix, coupling_matrix, aa_to_index = args
    wt_residue, index, mutant_residue, mutated_seq = mutation
    index_zero_based = index - 1  # Convert to 0-based index

    # Calculate the difference caused by the mutation
    delta_score = 0.0
    mutant_idx = aa_to_index[mutant_residue]
    original_idx = aa_to_index[wt_residue]
    num_residues = len(original_sequence_indices)

    for j in range(num_residues):
        if j == index_zero_based:
            continue
        coupling_score = coupling_matrix[index, j + 1]  # 1-based indexing
        if coupling_score != 0:
            aa_j_idx = original_sequence_indices[j]
            delta_score += (mj_matrix[mutant_idx, aa_j_idx] - mj_matrix[original_idx, aa_j_idx]) * coupling_score

    mutated_score = original_score + delta_score
    mutation_label = f"{wt_residue}{index}{mutant_residue}"
    return mutation_label, mutated_score

def save_mj_matrix(mj_matrix, index_to_aa, filename):
    """
    Save the MJ matrix to a file in a readable format.
    """
    with open(filename, 'w') as f:
        f.write("AA1-AA2\tScore\n")
        for i in range(len(index_to_aa)):
            for j in range(len(index_to_aa)):
                if i <= j:  # Avoid duplicates
                    aa1, aa2 = index_to_aa[i], index_to_aa[j]
                    score = mj_matrix[i, j]
                    if score != 0:
                        f.write(f"{aa1}-{aa2}\t{score}\n")

def save_coupling_matrix(coupling_matrix, filename):
    """
    Save the coupling score matrix as a sorted list to a file.
    """
    with open(filename, 'w') as f:
        f.write("Residues\tScore\n")
        num_residues = coupling_matrix.shape[0] - 1
        for i in range(1, num_residues + 1):
            for j in range(i, num_residues + 1):
                score = coupling_matrix[i, j]
                if score != 0:
                    f.write(f"{i}-{j}\t{score}\n")

def save_weighted_scores(scores, original_score, filename):
    """
    Save the original and mutated MJ scores with their differences to a file.
    """
    with open(filename, 'w') as f:
        f.write("Label\tScore\tDifference\n")
        # Define a custom sort key
        def sort_key(item):
            label, score = item
            if label == 'wt':
                return (0, 0, '')  # Ensure 'wt' is first
            else:
                # Extract position and mutant amino acid
                import re
                match = re.match(r'^([A-Z])(\d+)([A-Z])$', label)
                if match:
                    pos = int(match.group(2))
                    mutant_aa = match.group(3)
                    return (1, pos, mutant_aa)
                else:
                    return (2, 0, label)  # Place any malformed labels at the end

        for label, score in sorted(scores.items(), key=sort_key):
            difference = (original_score - score) if label != "wt" else 0
            f.write(f"{label}\t{score}\t{difference}\n")

def main(fasta_file, coupling_file, output_dir):
    """
    Main function to run the analysis.
    """
    # Load the sequence
    sequence = load_fasta(fasta_file)
    print(f"Original Sequence: {sequence}")
    num_residues = len(sequence)

    # Convert sequence to a list for mutation
    sequence_list = list(sequence)

    # Convert sequence to indices for faster processing
    try:
        sequence_indices = [aa_to_index[aa] for aa in sequence_list]
    except KeyError as e:
        raise ValueError(f"Invalid amino acid in sequence: {e}")

    # Load coupling scores
    coupling_matrix = load_coupling_scores(coupling_file, num_residues)

    # Save the coupling score matrix as a sorted list
    coupling_output_path = os.path.join(output_dir, "coupling_scores_matrix.txt")
    save_coupling_matrix(coupling_matrix, coupling_output_path)
    print(f"Coupling scores saved to {coupling_output_path}")

    # Save the MJ matrix
    mj_output_path = os.path.join(output_dir, "mj_matrix.txt")
    save_mj_matrix(mj_matrix, index_to_aa, mj_output_path)
    print(f"MJ matrix saved to {mj_output_path}")

    # Calculate the original score
    original_score = calculate_mj_score(sequence_indices, mj_matrix, coupling_matrix)
    print(f"Original MJ Score: {original_score}")

    # Generate all possible single amino acid mutations
    mutations = generate_mutations(sequence_list, amino_acids)
    print(f"Total Mutations: {len(mutations)}")

    # Prepare arguments for multiprocessing
    args_list = []
    for mutation in mutations:
        args = (sequence_indices, original_score, mutation, mj_matrix, coupling_matrix, aa_to_index)
        args_list.append(args)

    # Calculate the mutation scores in parallel
    scores = {"wt": original_score}
    print("Calculating mutation scores in parallel...")
    with ProcessPoolExecutor() as executor:
        # Submit all mutation calculations
        futures = [executor.submit(calculate_single_mutation_score, args) for args in args_list]
        for future in as_completed(futures):
            try:
                mutation_label, mutated_score = future.result()
                scores[mutation_label] = mutated_score
            except Exception as e:
                print(f"Error processing a mutation: {e}")

    # Save the weighted MJ scores
    stability_output_path = os.path.join(output_dir, "stability_scores.txt")
    save_weighted_scores(scores, original_score, stability_output_path)
    print(f"Stability scores saved to {stability_output_path}")

    # Print a sample of mutation scores
    print("\nSample Mutation Scores:")
    count = 0
    for label, score in scores.items():
        difference = original_score - score if label != "wt" else 0
        print(f"{label}: {score}, Difference: {difference}")
        count += 1
        if count >= 25:  
            break
    if len(scores) > 26:
        print(f"...and {len(scores) - 26} more mutations.")

if __name__ == "__main__":
    # Define file paths (modify these paths as needed)
    fasta_file = ""  # Change this to your FASTA file path
    coupling_file = ""  # Change this to your EVcouplings coupling scores CSV file path
    output_dir = ""  # Change this to your desired output directory

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run the main function
    main(fasta_file, coupling_file, output_dir)