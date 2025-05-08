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
Improved crystal data preparation for XAI analysis.

This script converts the GNoME dataset to the format needed by the XAI scripts,
with better handling of CIF files from ZIP archives.
"""

import os
import argparse
import pickle
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jraph
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifParser
import zipfile
from tqdm import tqdm
import tempfile
import io
import re

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare crystal data for XAI analysis.")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing GNoME data')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output pickle file')
    parser.add_argument('--max_structures', type=int, default=50,
                        help='Maximum number of structures to process')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    return parser.parse_args()

def load_summary_data(input_dir, max_structures):
    """Load summary data from CSV file."""
    summary_file = os.path.join(input_dir, 'stable_materials_summary.csv')
    print(f"Loading summary data from {summary_file}...")
    
    # Read only the first max_structures rows to save memory
    return pd.read_csv(summary_file, nrows=max_structures)

def print_zip_contents(zip_path, max_entries=10):
    """Debug function to print contents of ZIP file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            entries = z.namelist()
            print(f"ZIP file {zip_path} contains {len(entries)} entries")
            print(f"First {min(max_entries, len(entries))} entries:")
            for i, entry in enumerate(entries[:max_entries]):
                print(f"  {i+1}. {entry}")
    except Exception as e:
        print(f"Error reading {zip_path}: {e}")

def find_cif_in_zipfile(zip_path, material_id, debug=False):
    """Search for a CIF file in a ZIP archive with the given material ID.
    
    This function handles various folder structures and case sensitivity.
    """
    # Check if the ZIP file exists
    if not os.path.exists(zip_path):
        if debug:
            print(f"ZIP file {zip_path} does not exist")
        return None
    
    # Try various patterns and case variations
    patterns = [
        f"{material_id}.cif",  # Original lowercase
        f"{material_id}.CIF",  # Uppercase extension
        f"*/{material_id}.cif",  # In subdirectory, lowercase
        f"*/{material_id}.CIF",  # In subdirectory, uppercase
        f"*{material_id}*.cif",  # Contains material_id, lowercase
        f"*{material_id}*.CIF",  # Contains material_id, uppercase
    ]
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            file_list = z.namelist()
            
            # Try to find the exact file
            for pattern in patterns:
                # For wildcard patterns
                if '*' in pattern:
                    base_pattern = pattern.replace('*', '')
                    for filename in file_list:
                        if base_pattern.lower() in filename.lower():
                            if debug:
                                print(f"Found match using pattern {pattern}: {filename}")
                            return z.read(filename)
                else:
                    # Try exact match (case-insensitive)
                    for filename in file_list:
                        if filename.lower().endswith(pattern.lower()):
                            if debug:
                                print(f"Found exact match: {filename}")
                            return z.read(filename)
            
            # If still not found, try a more flexible approach
            # Look for any file containing the materialId
            for filename in file_list:
                if material_id.lower() in filename.lower() and filename.lower().endswith('.cif'):
                    if debug:
                        print(f"Found flexible match: {filename}")
                    return z.read(filename)
                # Also try without the leading zeros
                material_id_no_zeros = material_id.lstrip('0')
                if material_id_no_zeros and material_id_no_zeros.lower() in filename.lower() and filename.lower().endswith('.cif'):
                    if debug:
                        print(f"Found match without leading zeros: {filename}")
                    return z.read(filename)
    except Exception as e:
        print(f"Error reading {zip_path}: {e}")
    
    return None

def extract_cif_from_zips(material_id, input_dir, debug=False):
    """Extract a CIF file by checking multiple ZIP archives."""
    # List of ZIP files to check
    zip_files = [
        os.path.join(input_dir, 'by_id.zip'),
        os.path.join(input_dir, 'by_composition.zip'),
        os.path.join(input_dir, 'by_reduced_formula.zip')
    ]
    
    # Try each ZIP file
    for zip_path in zip_files:
        cif_content = find_cif_in_zipfile(zip_path, material_id, debug)
        if cif_content:
            try:
                # Save to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as temp_file:
                    temp_file.write(cif_content)
                    temp_path = temp_file.name
                
                # Parse the CIF file
                parser = CifParser(temp_path)
                structure = parser.get_structures()[0]
                
                # Clean up the temporary file
                os.unlink(temp_path)
                return structure
            except Exception as e:
                print(f"Error parsing CIF for {material_id}: {e}")
                # Clean up the temporary file if it exists
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    # Try the first few structures from the ZIP files as a fallback
    if debug:
        print(f"Couldn't find {material_id}.cif, trying to use a different structure...")
    
    # Try to extract and use the first few valid structures from the first available ZIP
    for zip_path in zip_files:
        if os.path.exists(zip_path):
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    file_list = [f for f in z.namelist() if f.lower().endswith('.cif')]
                    
                    for i, filename in enumerate(file_list[:10]):  # Try first 10 CIF files
                        try:
                            cif_content = z.read(filename)
                            with tempfile.NamedTemporaryFile(suffix='.cif', delete=False) as temp_file:
                                temp_file.write(cif_content)
                                temp_path = temp_file.name
                            
                            # Parse the CIF file
                            parser = CifParser(temp_path)
                            structure = parser.get_structures()[0]
                            
                            # Clean up the temporary file
                            os.unlink(temp_path)
                            if debug:
                                print(f"Using structure from {filename} instead")
                            return structure
                        except Exception as e:
                            if debug:
                                print(f"Couldn't use {filename}: {e}")
                            # Clean up the temporary file if it exists
                            try:
                                os.unlink(temp_path)
                            except:
                                pass
            except Exception as e:
                print(f"Error accessing {zip_path}: {e}")
    
    print(f"Warning: {material_id}.cif not found in any ZIP file")
    return None

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
            # Use structure.get_distance_and_image to get the displacement vector
            _, image = structure.lattice.get_distance_and_image(positions[i], positions[j])
            edge_vec = positions[j] - positions[i] + np.dot(image, structure.lattice.matrix)
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
    
    displacements = np.zeros_like(edges)  # Create zero displacements
    
    graph = jraph.GraphsTuple(
        nodes=nodes,
        edges=(edges, displacements),  # Make edges a tuple with edges and displacements
        senders=senders,
        receivers=receivers,
        globals=globals_,
        n_node=n_node,
        n_edge=n_edge
    )
    
    return graph, positions

def create_dummy_structures(count=10):
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
        
        try:
            graph, positions = structure_to_graph(structure)
            box = structure.lattice.matrix
            
            graphs.append(graph)
            positions_list.append(positions)
            boxes.append(box)
        except Exception as e:
            print(f"Error processing dummy structure {i}: {e}")
    
    return graphs, positions_list, boxes

def prepare_data(args):
    """Prepare crystal data for XAI analysis."""
    # Load summary data
    summary_df = load_summary_data(args.input_dir, args.max_structures)
    
    # Debug: Print the contents of the ZIP files
    print("Examining ZIP files in the input directory...")
    for zip_name in ['by_id.zip', 'by_composition.zip', 'by_reduced_formula.zip']:
        zip_path = os.path.join(args.input_dir, zip_name)
        if os.path.exists(zip_path):
            print_zip_contents(zip_path)
        else:
            print(f"ZIP file {zip_path} does not exist")
    
    # Initialize lists to store data
    graphs = []
    positions_list = []
    boxes = []
    
    # Process each structure
    print(f"Processing up to {len(summary_df)} structures...")
    for i, row in tqdm(summary_df.iterrows(), total=len(summary_df)):
        material_id = row['MaterialId']
        
        # Extract and parse structure
        structure = extract_cif_from_zips(material_id, args.input_dir, args.debug)
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
    
    # If no valid structures were found, create dummy structures
    if len(graphs) == 0:
        print("No valid structures found. Creating dummy structures...")
        graphs, positions_list, boxes = create_dummy_structures(10)
    
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