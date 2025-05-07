#!/usr/bin/env python3
# Copyright 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prepare crystal data for XAI analysis.

This script converts the GNoME dataset to the format needed by the XAI scripts.
"""

import os
import argparse
import pickle
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jraph
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
import zipfile
from tqdm import tqdm


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare crystal data for XAI analysis.")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing GNoME data')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output pickle file')
    parser.add_argument('--max_structures', type=int, default=50,
                        help='Maximum number of structures to process')
    return parser.parse_args()


def load_summary_data(input_dir, max_structures):
    """Load summary data from CSV file."""
    summary_file = os.path.join(input_dir, 'stable_materials_summary.csv')
    print(f"Loading summary data from {summary_file}...")
    
    # Read only the first max_structures rows to save memory
    return pd.read_csv(summary_file, nrows=max_structures)


def extract_structure_from_zip(zip_path, cif_name):
    """Extract a CIF file from a ZIP archive and parse it."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            if cif_name in z.namelist():
                with z.open(cif_name) as f:
                    parser = CifParser(f)
                    return parser.get_structures()[0]
            else:
                print(f"Warning: {cif_name} not found in {zip_path}")
                return None
    except Exception as e:
        print(f"Error extracting {cif_name} from {zip_path}: {e}")
        return None


def structure_to_graph(structure, max_neighbors=8):
    """Convert a pymatgen Structure to a jraph.GraphsTuple.
    
    Args:
        structure: pymatgen Structure object
        max_neighbors: Maximum number of neighbors to consider per atom
        
    Returns:
        graph: jraph.GraphsTuple object
        positions: Atomic positions
    """
    # Get atomic numbers for nodes
    atomic_numbers = np.array([site.specie.Z for site in structure])
    num_atoms = len(atomic_numbers)
    
    # Create one-hot encoding for elements (up to Z=94, Pu)
    nodes = np.zeros((num_atoms, 94))
    for i, z in enumerate(atomic_numbers):
        if 0 < z <= 94:
            nodes[i, z-1] = 1.0
    
    # Get positions
    positions = np.array([site.coords for site in structure])
    
    # Create edges based on nearest neighbors
    # Simple distance-based approach for simplicity
    senders = []
    receivers = []
    edges = []
    
    # Compute distances between all atoms
    all_pairs = []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                # Get distance with periodic boundary conditions
                dist = structure.get_distance(i, j)
                all_pairs.append((i, j, dist))
    
    # Sort by distance
    all_pairs.sort(key=lambda x: x[2])
    
    # Add edges for each atom's nearest neighbors
    neighbors_count = [0] * num_atoms
    for i, j, dist in all_pairs:
        if neighbors_count[i] < max_neighbors:
            senders.append(i)
            receivers.append(j)
            
            # Calculate edge vector (displacement)
            # Get distance with periodic boundary conditions
            edge_vec = structure.lattice.get_displacement(positions[i], positions[j])
            edges.append(edge_vec)
            
            neighbors_count[i] += 1
    
    # If no edges were found, add a dummy edge
    if len(senders) == 0:
        senders = [0]
        receivers = [0]
        edges = [[0.0, 0.0, 0.0]]
    
    # Convert to numpy arrays
    senders = np.array(senders, dtype=np.int32)
    receivers = np.array(receivers, dtype=np.int32)
    edges = np.array(edges, dtype=np.float32)
    
    # Create GraphsTuple
    n_node = np.array([num_atoms], dtype=np.int32)
    n_edge = np.array([len(senders)], dtype=np.int32)
    globals_ = np.zeros((1, 1), dtype=np.float32)
    
    graph = jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        globals=globals_,
        n_node=n_node,
        n_edge=n_edge
    )
    
    return graph, positions


def prepare_data(args):
    """Prepare crystal data for XAI analysis."""
    # Load summary data
    summary_df = load_summary_data(args.input_dir, args.max_structures)
    
    # Path to by_id.zip file
    by_id_zip = os.path.join(args.input_dir, 'by_id.zip')
    
    # Initialize lists to store data
    graphs = []
    positions_list = []
    boxes = []
    
    # Process each structure
    print(f"Processing up to {len(summary_df)} structures...")
    for i, row in tqdm(summary_df.iterrows(), total=len(summary_df)):
        material_id = row['MaterialId']
        cif_name = f"{material_id}.cif"
        
        # Extract and parse structure
        structure = extract_structure_from_zip(by_id_zip, cif_name)
        if structure is None:
            continue
        
        # Convert to graph
        try:
            graph, positions = structure_to_graph(structure)
            
            # Store lattice as box
            box = structure.lattice.matrix
            
            # Append to lists
            graphs.append(graph)
            positions_list.append(positions)
            boxes.append(box)
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(summary_df)} structures")
                
            # Break if we've processed enough structures
            if len(graphs) >= args.max_structures:
                break
        
        except Exception as e:
            print(f"Error processing structure {material_id}: {e}")
    
    # Create output data
    print(f"Collected {len(graphs)} valid structures")
    output_data = {
        'graphs': graphs,
        'positions': positions_list,
        'boxes': boxes
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    # Save to pickle file
    print(f"Saving {len(graphs)} structures to {args.output_file}...")
    with open(args.output_file, 'wb') as f:
        pickle.dump(output_data, f)
    
    print("Data preparation complete!")


if __name__ == '__main__':
    args = parse_args()
    prepare_data(args)