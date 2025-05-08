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
Create dummy crystal structures for testing.

This script generates simple crystal structures and saves them in the format
expected by the XAI analysis scripts.
"""

import os
import sys
import pickle
import numpy as np
import jraph
from pymatgen.core import Lattice, Structure

def structure_to_graph(structure, max_neighbors=8):
    """Convert a pymatgen Structure to a jraph.GraphsTuple."""
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
    senders = []
    receivers = []
    edges = []
    
    # Compute all pairs of distances
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

def create_dummy_structures(output_file, count=10):
    """Create dummy crystal structures for testing."""
    graphs = []
    positions_list = []
    boxes = []
    
    print(f"Creating {count} dummy structures...")
    
    # Create some varied dummy structures
    structure_templates = [
        # Simple cubic
        (Lattice.cubic(4.0), 
         ["C", "C", "C", "C", "O", "O", "H", "H"], 
         [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0],
          [0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0.1, 0.1, 0.1], [0.9, 0.9, 0.9]]),
          
        # BCC
        (Lattice.cubic(3.6), 
         ["Fe", "Fe", "Fe", "Fe", "Fe", "Fe", "Fe", "Fe", "Fe"], 
         [[0, 0, 0], [0.5, 0.5, 0.5], [0, 0.5, 0.5], [0.5, 0, 0.5], 
          [0.5, 0.5, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75], 
          [0.25, 0.75, 0.75], [0.75, 0.25, 0.25]]),
          
        # FCC
        (Lattice.cubic(3.8), 
         ["Al", "Al", "Al", "Al"], 
         [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]),
         
        # Hexagonal
        (Lattice.hexagonal(3.2, 5.2), 
         ["Ti", "Ti", "Ti", "Ti", "O", "O", "O", "O", "O", "O"], 
         [[0, 0, 0], [1/3, 2/3, 0.25], [2/3, 1/3, 0.75], [0, 0, 0.5],
          [0.3, 0, 0.1], [0.7, 0, 0.1], [0, 0.3, 0.6], [0, 0.7, 0.6],
          [1/3, 2/3, 0.85], [2/3, 1/3, 0.35]]),
          
        # Perovskite-like
        (Lattice.cubic(4.2), 
         ["Ca", "Ti", "O", "O", "O"], 
         [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    ]
    
    # Generate structures from templates and variations
    for i in range(count):
        template_idx = i % len(structure_templates)
        lattice, species, coords = structure_templates[template_idx]
        
        # Add small random variations to avoid identical structures
        coords_with_noise = [[c[0] + np.random.normal(0, 0.01),
                              c[1] + np.random.normal(0, 0.01),
                              c[2] + np.random.normal(0, 0.01)] for c in coords]
        
        structure = Structure(lattice, species, coords_with_noise)
        
        graph, positions = structure_to_graph(structure)
        box = structure.lattice.matrix
        
        graphs.append(graph)
        positions_list.append(positions)
        boxes.append(box)
    
    # Create output data
    output_data = {
        'graphs': graphs,
        'positions': positions_list,
        'boxes': boxes
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Save to pickle file
    print(f"Saving {len(graphs)} structures to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"Created {len(graphs)} dummy structures in {output_file}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python create_dummy_structures.py output_file.pkl [count]")
        sys.exit(1)
    
    output_file = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    create_dummy_structures(output_file, count)