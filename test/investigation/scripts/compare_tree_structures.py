#!/usr/bin/env python3
"""
Compare CUDA and OpenCL tree structures to identify differences.

This script parses tree structure logs from CUDA and OpenCL implementations
and performs a comprehensive comparison to identify any divergences.
"""

import re
import sys
from typing import Dict, List, Tuple

def parse_cuda_structure(filename: str) -> Dict:
    """Parse CUDA tree structure log."""
    with open(filename, 'r') as f:
        content = f.read()

    data = {
        'summary': {},
        'nodes': [],
        'hybrid_array': [],
        'pivots': []
    }

    # Parse tree summary
    if m := re.search(r'Total nodes: (\d+)', content):
        data['summary']['total_nodes'] = int(m.group(1))
    if m := re.search(r'Parents: (\d+), Leaves: (\d+)', content):
        data['summary']['parents'] = int(m.group(1))
        data['summary']['leaves'] = int(m.group(2))
    if m := re.search(r'Trees: (\d+), Branching: (\d+)', content):
        data['summary']['trees'] = int(m.group(1))
        data['summary']['branching'] = int(m.group(2))
    if m := re.search(r'Hybrid array size: (\d+)', content):
        data['summary']['hybrid_size'] = int(m.group(1))

    # Parse nodes (child_start values)
    for match in re.finditer(r'Node\[\s*(\d+)\]:\s+child_start=\s*(-?\d+),\s+count=\s*(\d+),\s+(PARENT|LEAF)', content):
        node_id = int(match.group(1))
        child_start = int(match.group(2))
        count = int(match.group(3))
        node_type = match.group(4)
        data['nodes'].append({
            'id': node_id,
            'child_start': child_start,
            'count': count,
            'type': node_type
        })

    # Parse hybrid array values
    for match in re.finditer(r'hybrid\[\s*(\d+)\]\s*=\s*(-?\d+)', content):
        idx = int(match.group(1))
        value = int(match.group(2))
        data['hybrid_array'].append({'idx': idx, 'value': value})

    # Parse pivot descriptors (first 8 bytes)
    for match in re.finditer(r'Pivot\s+(\d+):\s+([\da-f\s]+)\(', content):
        pivot_id = int(match.group(1))
        bytes_hex = match.group(2).strip().split()[:8]  # First 8 bytes
        data['pivots'].append({'id': pivot_id, 'bytes': bytes_hex})

    return data

def parse_opencl_structure(filename: str) -> Dict:
    """Parse OpenCL tree structure log."""
    with open(filename, 'r') as f:
        content = f.read()

    data = {
        'summary': {},
        'nodes': [],
        'pivots': []
    }

    # Parse tree summary
    if m := re.search(r'Trees: (\d+), Branching: (\d+)', content):
        data['summary']['trees'] = int(m.group(1))
        data['summary']['branching'] = int(m.group(2))
    if m := re.search(r'First level children: (\d+)', content):
        data['summary']['first_level_children'] = int(m.group(1))
    if m := re.search(r'veclen: (\d+)', content):
        data['summary']['veclen'] = int(m.group(1))

    # Parse nodeIndex values (equivalent to CUDA child_start)
    for match in re.finditer(r'nodeIndex\[\s*(\d+)\]\s*=\s*(-?\d+)', content):
        idx = int(match.group(1))
        value = int(match.group(2))
        # Determine node type based on value (negative = leaf pointer in hybrid array)
        node_type = "PARENT" if value > 0 else "LEAF"
        data['nodes'].append({
            'id': idx,
            'child_start': value,
            'type': node_type
        })

    # Parse pivot descriptors (first 8 bytes)
    for match in re.finditer(r'Pivot\s+(\d+):\s+([\da-f\s]+)', content):
        pivot_id = int(match.group(1))
        bytes_hex = match.group(2).strip().split()[:8]  # First 8 bytes
        data['pivots'].append({'id': pivot_id, 'bytes': bytes_hex})

    return data

def compare_summaries(cuda_data: Dict, opencl_data: Dict) -> List[str]:
    """Compare tree summaries."""
    differences = []

    cuda_sum = cuda_data['summary']
    opencl_sum = opencl_data['summary']

    if cuda_sum.get('trees') != opencl_sum.get('trees'):
        differences.append(f"❌ Trees count: CUDA={cuda_sum.get('trees')}, OpenCL={opencl_sum.get('trees')}")
    else:
        differences.append(f"✅ Trees count: {cuda_sum.get('trees')} (MATCH)")

    if cuda_sum.get('branching') != opencl_sum.get('branching'):
        differences.append(f"❌ Branching: CUDA={cuda_sum.get('branching')}, OpenCL={opencl_sum.get('branching')}")
    else:
        differences.append(f"✅ Branching factor: {cuda_sum.get('branching')} (MATCH)")

    if 'total_nodes' in cuda_sum:
        differences.append(f"ℹ️  CUDA total nodes: {cuda_sum['total_nodes']} (parents={cuda_sum.get('parents')}, leaves={cuda_sum.get('leaves')})")

    if 'hybrid_size' in cuda_sum:
        differences.append(f"ℹ️  CUDA hybrid array size: {cuda_sum['hybrid_size']}")

    return differences

def compare_nodes(cuda_data: Dict, opencl_data: Dict) -> List[str]:
    """Compare node structures (child_start values)."""
    differences = []

    cuda_nodes = {n['id']: n for n in cuda_data['nodes']}
    opencl_nodes = {n['id']: n for n in opencl_data['nodes']}

    # Check all node IDs present in both
    all_ids = sorted(set(cuda_nodes.keys()) | set(opencl_nodes.keys()))

    matching = 0
    mismatched = 0
    cuda_only = 0
    opencl_only = 0

    for node_id in all_ids:
        cuda_node = cuda_nodes.get(node_id)
        opencl_node = opencl_nodes.get(node_id)

        if cuda_node and not opencl_node:
            cuda_only += 1
            if cuda_only <= 5:  # Show first 5
                differences.append(f"❌ Node {node_id}: Present in CUDA only (child_start={cuda_node['child_start']})")
        elif opencl_node and not cuda_node:
            opencl_only += 1
            if opencl_only <= 5:  # Show first 5
                differences.append(f"❌ Node {node_id}: Present in OpenCL only (child_start={opencl_node['child_start']})")
        elif cuda_node and opencl_node:
            if cuda_node['child_start'] != opencl_node['child_start']:
                mismatched += 1
                if mismatched <= 10:  # Show first 10 mismatches
                    differences.append(
                        f"❌ Node {node_id}: child_start mismatch: "
                        f"CUDA={cuda_node['child_start']} ({cuda_node['type']}), "
                        f"OpenCL={opencl_node['child_start']} ({opencl_node['type']})"
                    )
            else:
                matching += 1

    # Summary
    differences.insert(0, f"")
    differences.insert(0, f"ℹ️  Nodes only in OpenCL: {opencl_only}")
    differences.insert(0, f"ℹ️  Nodes only in CUDA: {cuda_only}")
    differences.insert(0, f"ℹ️  Mismatched nodes: {mismatched}")
    differences.insert(0, f"✅ Matching nodes: {matching}")

    if mismatched > 10:
        differences.append(f"   ... and {mismatched - 10} more mismatches")

    return differences

def compare_pivots(cuda_data: Dict, opencl_data: Dict) -> List[str]:
    """Compare pivot descriptors (first 8 bytes)."""
    differences = []

    cuda_pivots = {p['id']: p for p in cuda_data['pivots']}
    opencl_pivots = {p['id']: p for p in opencl_data['pivots']}

    all_ids = sorted(set(cuda_pivots.keys()) | set(opencl_pivots.keys()))

    matching = 0
    mismatched = 0

    for pivot_id in all_ids:
        cuda_pivot = cuda_pivots.get(pivot_id)
        opencl_pivot = opencl_pivots.get(pivot_id)

        if cuda_pivot and opencl_pivot:
            if cuda_pivot['bytes'] == opencl_pivot['bytes']:
                matching += 1
            else:
                mismatched += 1
                if mismatched <= 10:  # Show first 10 mismatches
                    cuda_hex = ' '.join(cuda_pivot['bytes'])
                    opencl_hex = ' '.join(opencl_pivot['bytes'])
                    differences.append(
                        f"❌ Pivot {pivot_id}: CUDA=[{cuda_hex}], OpenCL=[{opencl_hex}]"
                    )
        elif cuda_pivot and not opencl_pivot:
            if mismatched < 5:
                differences.append(f"❌ Pivot {pivot_id}: Present in CUDA only")
        elif opencl_pivot and not cuda_pivot:
            if mismatched < 5:
                differences.append(f"❌ Pivot {pivot_id}: Present in OpenCL only")

    # Summary
    differences.insert(0, f"")
    differences.insert(0, f"ℹ️  Mismatched pivots: {mismatched}")
    differences.insert(0, f"✅ Matching pivots: {matching}")

    if mismatched > 10:
        differences.append(f"   ... and {mismatched - 10} more mismatches")

    return differences

def main():
    cuda_file = "investigation/tree_structures/cuda_structure.log"
    opencl_file = "investigation/tree_structures/opencl_structure.log"
    output_file = "investigation/reports/tree_comparison_report.md"

    print("🔍 Parsing CUDA tree structure...")
    cuda_data = parse_cuda_structure(cuda_file)

    print("🔍 Parsing OpenCL tree structure...")
    opencl_data = parse_opencl_structure(opencl_file)

    print("📊 Comparing tree structures...")

    # Generate comparison report
    report = []
    report.append("# Tree Structure Comparison Report")
    report.append("")
    report.append("## Executive Summary")
    report.append("")
    report.append("Comparison of CUDA and OpenCL hierarchical tree structures.")
    report.append("")
    report.append("---")
    report.append("")

    # Summary comparison
    report.append("## 1. Tree Summary Comparison")
    report.append("")
    summary_diffs = compare_summaries(cuda_data, opencl_data)
    report.extend(summary_diffs)
    report.append("")
    report.append("---")
    report.append("")

    # Node comparison
    report.append("## 2. Node Structure Comparison (child_start values)")
    report.append("")
    report.append("Comparing CUDA `tree_nodes[i].child_start` with OpenCL `nodeIndex[i]`")
    report.append("")
    node_diffs = compare_nodes(cuda_data, opencl_data)
    report.extend(node_diffs)
    report.append("")
    report.append("---")
    report.append("")

    # Pivot comparison
    report.append("## 3. Pivot Descriptor Comparison (first 8 bytes)")
    report.append("")
    pivot_diffs = compare_pivots(cuda_data, opencl_data)
    report.extend(pivot_diffs)
    report.append("")
    report.append("---")
    report.append("")

    # Conclusion
    report.append("## 4. Conclusion")
    report.append("")

    # Determine if structures match
    total_node_mismatches = sum(1 for line in node_diffs if line.startswith("❌"))
    total_pivot_mismatches = sum(1 for line in pivot_diffs if line.startswith("❌"))

    if total_node_mismatches == 0 and total_pivot_mismatches == 0:
        report.append("### ✅ **TREE STRUCTURES ARE IDENTICAL**")
        report.append("")
        report.append("CUDA and OpenCL build identical tree structures:")
        report.append("- All node child_start values match")
        report.append("- All pivot descriptors match")
        report.append("")
        report.append("**Implication**: The precision difference (CUDA 89.6% vs OpenCL 97.2%) is NOT caused by different tree structures.")
        report.append("The divergence must occur during the search phase, not tree building.")
    else:
        report.append("### ❌ **TREE STRUCTURES DIFFER**")
        report.append("")
        report.append(f"- Node mismatches: {total_node_mismatches}")
        report.append(f"- Pivot mismatches: {total_pivot_mismatches}")
        report.append("")
        report.append("**Implication**: Different tree structures will lead to different search results.")
        report.append("This could explain the precision gap (CUDA 89.6% vs OpenCL 97.2%).")

    report.append("")
    report.append("---")
    report.append("")
    report.append("**Generated**: 2025-11-18")
    report.append(f"**CUDA nodes analyzed**: {len(cuda_data['nodes'])}")
    report.append(f"**OpenCL nodes analyzed**: {len(opencl_data['nodes'])}")
    report.append(f"**CUDA pivots analyzed**: {len(cuda_data['pivots'])}")
    report.append(f"**OpenCL pivots analyzed**: {len(opencl_data['pivots'])}")

    # Write report
    report_text = '\n'.join(report)
    with open(output_file, 'w') as f:
        f.write(report_text)

    print(f"✅ Report written to {output_file}")
    print("")
    print("=" * 70)
    print(report_text)
    print("=" * 70)

    return 0 if (total_node_mismatches == 0 and total_pivot_mismatches == 0) else 1

if __name__ == "__main__":
    sys.exit(main())
