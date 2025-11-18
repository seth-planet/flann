# CUDA Cooperative Kernel Regression Analysis

**Date**: 2025-11-18
**Finding**: Cooperative kernel migration introduced **3.2% precision regression**

---

## Precision Comparison

| Implementation | Precision | File | Status |
|----------------|-----------|------|--------|
| **Single-threaded CUDA** | 92.8% (0.928) | `hierarchical_search_kernel.cuh` | ✅ Good (but not used) |
| **Cooperative CUDA** | 89.6% (0.896) | `hierarchical_search_cooperative.cuh` | ❌ **Currently active (REGRESSION!)** |
| **OpenCL (target)** | 97.2% (0.972) | `nn_opencl_index.h` | ✅ Best |

---

## Regression Summary

**What happened**: The cooperative kernel migration (to match OpenCL's parallel architecture) **reduced precision** from 92.8% → 89.6%

**Gap analysis**:
- Single-threaded CUDA → OpenCL: **4.4% gap** (acceptable, different architecture)
- Cooperative CUDA → OpenCL: **7.6% gap** (unacceptable, should match since same architecture)
- **Regression**: Cooperative is **3.2% worse** than single-threaded

---

## Code Locations

### Single-Threaded Kernel (92.8% precision)
**File**: `/home/seth/pl/flann_ocl/flann/src/cpp/flann/algorithms/cuda/kernels/hierarchical_search_kernel.cuh`

**Architecture**:
- 1 thread per query
- Private heap per thread (no synchronization needed)
- Linear heap insertion (simple)
- No bitonic sort (uses simpler heap management)

**Launch**: `launch_hierarchical_search()` (line 55 of hierarchical_cuda_index.h)

### Cooperative Kernel (89.6% precision - CURRENTLY ACTIVE)
**File**: `/home/seth/pl/flann_ocl/flann/src/cpp/flann/algorithms/cuda/kernels/hierarchical_search_cooperative.cuh`

**Architecture**:
- 128 threads cooperate on 1 query (matches OpenCL)
- Shared heap across all threads
- Barrier synchronization required
- Bitonic sort for heap management

**Launch**: `launch_hierarchical_search_cooperative()` (line 75, **active at line 992**)

---

## Query 0 Results Comparison

### Single-Threaded CUDA (92.8%)
**Need to test** - switch back to single-threaded kernel and capture results

### Cooperative CUDA (89.6%)
```
Got:      [53249, 91202, 5321]
Distances: [34, 37, 38]
```
**Issue**: Finding ID=5321 @ distance=38 (wrong)

### Expected (Ground Truth)
```
Expected: [53249, 72652, 74891]
Distances: [34, 37, 37]
```

### OpenCL (97.2%)
```
Got:      [53249, 91202, 72652]
Distances: [34, 37, 37]
```
**Analysis**: Finds 2/3 correct neighbors

---

## Root Cause Hypotheses

### Why is Cooperative WORSE than Single-Threaded?

#### Hypothesis #1: Bitonic Sort Bug ⭐⭐⭐⭐⭐
**Theory**: The parallel bitonic sort has a bug that single-threaded heap doesn't have

**Evidence**:
- Single-threaded uses simpler heap management (no bitonic sort)
- Cooperative uses complex bitonic sort
- Already fixed one bitonic sort bug (bounds check), but precision didn't improve

**Test**: Compare heap states between single-threaded and cooperative

#### Hypothesis #2: Shared Memory Race Condition ⭐⭐⭐⭐
**Theory**: Despite barriers, there's still a race condition in shared heap access

**Evidence**:
- Cooperative uses shared memory with 128 threads
- Single-threaded has no synchronization issues
- Complex interaction between atomic ops and barriers

**Test**: Add extra barriers, check if precision improves

#### Hypothesis #3: Different Execution Order ⭐⭐⭐
**Theory**: Thread scheduling causes different leaf exploration order

**Evidence**:
- Cooperative: threads fetch dataset indices via `atomicAdd`
- Single-threaded: linear traversal, deterministic order
- Different order could lead to different heap evictions

**Test**: Log exact dataset index order examined

#### Hypothesis #4: Heap Capacity Effective Reduction ⭐⭐⭐
**Theory**: Cooperative's parallel insertion reduces effective heap capacity

**Evidence**:
- Both use 256-element heap
- But cooperative has 128 threads competing for heap space
- Potential conflicts/overwrites during concurrent insertion

**Test**: Increase heap size for cooperative, check precision

---

## Why is Cooperative NOT Matching OpenCL?

Both cooperative CUDA and OpenCL use **identical architecture**:
- 128 threads per query
- Shared heap (256 elements)
- Bitonic sort
- Same synchronization points

Yet cooperative CUDA (89.6%) is 7.6% worse than OpenCL (97.2%).

**Possible causes**:
1. **Subtle algorithmic difference** we haven't found yet
2. **Execution-level difference** (atomic operation ordering, thread scheduling)
3. **Compiler difference** (NVCC vs OpenCL compiler optimizations)
4. **Bug in CUDA implementation** of the cooperative algorithm

---

## Investigation Plan to Fix Regression

### Phase 1: Direct Comparison (Single vs Cooperative)

**Goal**: Understand what changed that broke precision

**Method**:
1. Temporarily switch back to single-threaded kernel (comment line 992, use `launch_hierarchical_search`)
2. Run test, capture Query 0 results
3. Compare with cooperative results
4. Identify which neighbors differ

**Expected findings**:
- If single-threaded finds [53249, 91202, **X**] where X is at distance 37
- And cooperative finds [53249, 91202, **5321**] where 5321 is at distance 38
- Then cooperative is evicting the correct neighbor

### Phase 2: Add Comprehensive Logging

**Goal**: Trace execution to find divergence

**Logging points**:
1. After find_nodes (which leaf nodes selected)
2. During find_leaves (which dataset indices examined)
3. Before/after bitonic sort (heap state changes)
4. After duplicate removal (what was marked)

**Create logs**:
- `single_threaded_trace_query0.log`
- `cooperative_trace_query0.log`

### Phase 3: Side-by-Side Analysis

**Goal**: Find exact point where executions diverge

**Compare**:
- Leaf nodes explored (should match)
- Dataset indices examined (should match)
- Heap states after sorting (MAY differ)
- Final results (differ - we know this)

**Identify**: First point where heap states diverge

### Phase 4: Fix the Bug

Based on divergence point:

**Scenario A: Different leaves explored**
→ Bug in `find_nodes_cooperative`
→ Fix: Correct node selection logic

**Scenario B: Different dataset points examined**
→ Bug in `find_leaves_cooperative`
→ Fix: Correct index calculation or atomic operations

**Scenario C: Same inputs, different heap**
→ Bug in bitonic sort or heap insertion
→ Fix: Correct heap management

### Phase 5: Validate Fix

**Success criteria**:
1. Cooperative precision >= 92.8% (matches or exceeds single-threaded)
2. Cooperative precision >= 97% (matches OpenCL target)
3. No performance regression
4. Deterministic results

---

## Immediate Next Steps

### Step 1: Test Single-Threaded Kernel (5 minutes)
```cpp
// In hierarchical_cuda_index.h line 992, change:
// bool success = launch_hierarchical_search_cooperative(
// TO:
bool success = launch_hierarchical_search(
```
- Rebuild and test
- Verify 92.8% precision
- Capture Query 0 results

### Step 2: Compare Results (10 minutes)
- Document what single-threaded finds vs cooperative
- Identify pattern of differences
- Hypothesize root cause

### Step 3: Add Logging Infrastructure (30 minutes)
- Host-side logging (not kernel printf)
- Copy heap to CPU at checkpoints
- Generate trace files for comparison

### Step 4: Root Cause Analysis (1-2 hours)
- Compare traces
- Find divergence point
- Diagnose bug

### Step 5: Implement Fix (varies)
- Based on root cause
- Test and validate
- Achieve >= 97% precision

---

## Expected Timeline

| Phase | Time | Outcome |
|-------|------|---------|
| Test single-threaded | 5 min | Confirm 92.8% precision |
| Compare results | 10 min | Identify neighbor differences |
| Add logging | 30 min | Infrastructure ready |
| Trace comparison | 1 hour | Find divergence point |
| Root cause analysis | 1 hour | Diagnose bug |
| Implement fix | 1-3 hours | Fix cooperative kernel |
| Validation | 30 min | Verify >= 97% precision |
| **TOTAL** | **4-6 hours** | **Cooperative matches OpenCL** |

---

## Success Metrics

✅ **Cooperative precision >= 97%** (matches OpenCL)
✅ **No regression** from single-threaded (>= 92.8%)
✅ **Deterministic results** (same output every run)
✅ **Performance maintained** (~0.013s search time)
✅ **Root cause documented** (comprehensive report)

---

## Files to Review

### Cooperative Kernel (needs fixing)
- `src/cpp/flann/algorithms/cuda/kernels/hierarchical_search_cooperative.cuh`
  - `find_nodes_cooperative` (lines 150-225)
  - `find_leaves_cooperative` (lines 248-340)
  - `store_results` (lines 360-402)
  - `bitonic_merge` in `bitonic_sort.cuh` (lines 29-79)

### Single-Threaded Kernel (working reference)
- `src/cpp/flann/algorithms/cuda/kernels/hierarchical_search_kernel.cuh`
  - Study heap management approach
  - Compare with cooperative version
  - Identify what works better

### Host Code
- `src/cpp/flann/algorithms/cuda/hierarchical_cuda_index.h`
  - Line 992: Switch between kernels for testing
  - Add logging infrastructure

---

## Conclusion

**Current Status**: Cooperative kernel has a **3.2% regression** vs single-threaded

**Root Cause**: Unknown (requires investigation)

**Fix Strategy**:
1. Compare single-threaded vs cooperative execution
2. Find where they diverge
3. Fix the bug in cooperative kernel
4. Achieve 97%+ precision

**Priority**: HIGH - This regression must be fixed before deployment

**Next Action**: Switch to single-threaded kernel to establish baseline, then compare

---

**Report Generated**: 2025-11-18
**Status**: Regression identified, fix in progress
**Target**: 97%+ precision for cooperative kernel
