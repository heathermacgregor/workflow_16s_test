# workflow_16s/upstream/sequences/utils.py

import gzip
from pathlib import Path
from typing import Union
from Bio import SeqIO


def get_all_values(dictionary):
    values = []
    for value in dictionary.values():
        # Check if the value is a nested dictionary
        if isinstance(value, dict): values.extend(get_all_values(value))  
        else: values.append(value)  
    return values

def import_seqs_fasta(fasta_path: Union[str, Path]):
    seqs = dict(zip(
        (record.id for record in SeqIO.parse(fasta_path, "fasta")), 
        (str(record.seq) for record in SeqIO.parse(fasta_path, "fasta"))
    ))
    return seqs

def fastq_gz_to_fasta(fastq_file: Union[str, Path], n_sequences: int = 0) -> Path:
    """
    Converts a FASTQ.GZ file to a FASTA file.

    Args:
        fastq_file:  Path to the input FASTQ.GZ file.
        n_sequences: Maximum number of sequences to convert.

    Returns:
        fasta_file: Path to the created FASTA file.
    """
    fastq_file = Path(fastq_file)
    fasta_file = fastq_file.with_suffix("").with_suffix(".fasta")
    
    try:
        with gzip.open(fastq_file, "rt") as fastq, open(fasta_file, "w") as fasta:
            seq_count = 0
            for i, line in enumerate(fastq):
                if i % 4 == 0:  # Sequence identifier
                    if not line.startswith("@"):
                        raise ValueError(f"Invalid FASTQ format in file {fastq_file} at line {i + 1}")
                    fasta.write(">" + line[1:])  # Convert to FASTA header
                elif i % 4 == 1:  # Sequence
                    fasta.write(line)
                    seq_count += 1
                if n_sequences > 0 and seq_count >= n_sequences: break
        return fasta_file
    except Exception as e:
        raise RuntimeError(f"Error converting {fastq_file} to FASTA: {e}")
