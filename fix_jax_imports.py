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
Fix JAX tree imports in the codebase.

This script updates all JAX tree imports to use tree_util instead of tree.
"""

import os
import re

def update_tree_imports(directory):
    """Update all JAX tree imports to use tree_util instead of tree."""
    pattern1 = r'jax\.tree\.map'
    replace1 = 'jax.tree_util.tree_map'
    
    pattern2 = r'from jax import tree'
    replace2 = 'from jax import tree_util as tree'
    
    pattern3 = r'from jax import tree_util as tree_util'
    replace3 = 'from jax import tree_util'
    
    pattern4 = r'tree = jax\.tree_util'
    replace4 = 'tree = jax.tree_util'
    
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Check for duplicate "as tree_util" and fix manually
                    if 'tree_util as tree_util' in content:
                        content = content.replace('tree_util as tree_util', 'tree_util')
                    
                    # Apply replacements
                    new_content = re.sub(pattern1, replace1, content)
                    new_content = re.sub(pattern2, replace2, new_content)
                    new_content = re.sub(pattern3, replace3, new_content)
                    new_content = re.sub(pattern4, replace4, new_content)
                    
                    # Write back only if changes were made
                    if new_content != content:
                        print(f"Updating imports in {file_path}")
                        with open(file_path, 'w') as f:
                            f.write(new_content)
                        count += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    print(f"Updated JAX tree imports in {count} files")

if __name__ == "__main__":
    # Update imports in GNoME directory
    update_tree_imports("GNoME")
    # Update imports in xai directory
    update_tree_imports("xai")