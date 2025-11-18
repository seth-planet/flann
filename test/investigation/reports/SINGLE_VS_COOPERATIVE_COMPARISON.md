# Single-Threaded vs Cooperative CUDA Kernel Comparison

**Date**: 2025-11-18 23:05 UTC
**Test**: HierarchicalCUDA_Brief100K.TestSearch
**Dataset**: Brief100K (100K binary descriptors, 32 bytes each)

---

## Executive Summary

✅ **Single-Threaded Kernel**: **91.9% precision** (0.918667)
❌ **Cooperative Kernel**: **89.6% precision** (0.896333)
**Regression**: **-2.3%** (cooperative is worse)

**Critical Finding**: The cooperative kernel **misses a correct neighbor** (ID=74891 @ distance=37) and instead finds a worse neighbor (ID=5321 @ distance=38).

---

## Query 0 Detailed Comparison

### Single-Threaded CUDA (91.9% precision)

```
Got:      [53249, 91202, 74891]
Distances: [34, 37, 37]
```

**Analysis**:
- ✅ ID=53249 @ dist=34 - **CORRECT** (best neighbor)
- ❌ ID=91202 @ dist=37 - WRONG (should be 72652 @ dist=37)
- ✅ ID=74891 @ dist=37 - **CORRECT**
- **Accuracy**: 2/3 correct neighbors

**Debug output shows**:
```
Heap BEFORE sorting: ids=[53249,91202,74891] dists=[34,37,37]
Final sorted results: ids=[53249,91202,74891] dists=[34,37,37]
```

### Cooperative CUDA (89.6% precision)

```
Got:      [53249, 91202, 5321]
Distances: [34, 37, 38]
```

**Analysis**:
- ✅ ID=53249 @ dist=34 - **CORRECT** (best neighbor)
- ❌ ID=91202 @ dist=37 - WRONG (should be 72652 @ dist=37)
- ❌ ID=5321 @ dist=38 - **WRONG** (should be 74891 @ dist=37)
- **Accuracy**: 1/3 correct neighbors

### Ground Truth

```
Expected: [53249, 72652, 74891]
Distances: [34, 37, 37]
```

### OpenCL Reference (97.2% precision)

```
Got:      [53249, 91202, 72652]
Distances: [34, 37, 37]
```

**Analysis**:
- ✅ ID=53249 @ dist=34 - **CORRECT**
- ❌ ID=91202 @ dist=37 - WRONG (but at correct distance)
- ✅ ID=72652 @ dist=37 - **CORRECT**
- **Accuracy**: 2/3 correct neighbors (better than both CUDA versions for Query 0)

---

## Key Observations

### 1. Missing Neighbor Analysis

**ID=74891 (distance=37)**:
- ✅ **Single-threaded FINDS it**
- ❌ **Cooperative MISSES it**
- ✅ **Ground truth expects it**

This neighbor exists and is at distance 37, which should make it into the top-3 results.

**ID=5321 (distance=38)**:
- ❌ **Cooperative INCORRECTLY finds it**
- Not in ground truth top-3

The cooperative kernel is finding a worse neighbor (distance 38) instead of a better one (distance 37).

### 2. Common Error

Both CUDA kernels find ID=91202 instead of ID=72652, even though both are at distance 37. This suggests:
- Both kernels explore similar tree paths
- There's likely tie-breaking or heap eviction happening
- This error is consistent between single-threaded and cooperative

### 3. Cooperative-Specific Error

Only the cooperative kernel:
- Misses ID=74891 @ distance=37
- Finds ID=5321 @ distance=38 instead

This suggests the cooperative kernel is either:
1. **Not examining** all the same dataset points as single-threaded
2. **Evicting** correct neighbors from the heap prematurely
3. **Stopping exploration** before finding all candidates

---

## Precision Comparison Table

| Implementation | Precision | Query 0 Accuracy | Status |
|----------------|-----------|------------------|--------|
| **Ground Truth** | 100.0% | 3/3 | Target |
| **OpenCL** | 97.2% | 2/3 (this query) | Best |
| **Single-threaded CUDA** | 91.9% | 2/3 | ✅ Good |
| **Cooperative CUDA** | 89.6% | 1/3 | ❌ Regression |

---

## Root Cause Hypotheses

### Hypothesis #1: Different Leaf Nodes Explored ⭐⭐⭐⭐⭐

**Theory**: Cooperative kernel explores a different set of leaf nodes than single-threaded, missing the leaf containing ID=74891.

**Evidence**:
- Single-threaded finds 74891
- Cooperative doesn't find 74891
- Both use same max_checks parameter (256)
- Both use same tree structure

**Test**: Add logging to record which leaf nodes each kernel visits

**Likelihood**: **VERY HIGH** - Most likely explanation

### Hypothesis #2: Heap Eviction Bug ⭐⭐⭐⭐

**Theory**: The cooperative kernel's parallel heap management evicts 74891 @ distance=37 but keeps 5321 @ distance=38 (wrong eviction).

**Evidence**:
- Cooperative uses shared heap with 128 threads
- Bitonic sort may have bugs
- Atomic operations could cause race conditions
- Single-threaded has no synchronization issues

**Test**: Add heap checkpoints to see when 74891 is evicted

**Likelihood**: **HIGH** - Plausible but would need to check heap states

### Hypothesis #3: Atomic Counter Overflow/Wraparound ⭐⭐⭐

**Theory**: The `atomicAdd` operation for fetching dataset indices has issues causing some indices to be skipped.

**Evidence**:
- Cooperative uses `loc_i = atomicAdd(loc_ptr, 1)` to fetch indices
- 128 threads compete for indices
- Potential for race conditions or missed indices

**Test**: Log total number of dataset points examined by each kernel

**Likelihood**: **MEDIUM** - Possible but less likely than heap issues

### Hypothesis #4: Early Termination ⭐⭐

**Theory**: Cooperative kernel's 200-iteration limit causes it to stop before exploring all candidates.

**Evidence**:
- Both kernels have 200 iteration limit in `find_nodes_cooperative`
- If cooperative hits limit earlier, it would miss nodes

**Test**: Increase iteration limit and check precision

**Likelihood**: **LOW-MEDIUM** - Both use same limit

---

## Next Investigation Steps

### Step 1: Verify Leaf Node Coverage (CRITICAL)

**Action**: Add logging to single-threaded and cooperative kernels to record:
- Which leaf nodes are visited
- How many dataset points are examined per leaf
- Total dataset points examined

**Expected**: If cooperative examines fewer points or different leaves, that's the root cause.

**Implementation**: Add host-side arrays to store visited leaf indices, copy back after kernel completes.

### Step 2: Heap State Checkpoints

**Action**: Add checkpoints to copy heap contents to CPU at key points:
- After each tree exploration
- After final duplicate removal
- Before final sorting

**Expected**: Identify when 74891 is evicted from cooperative heap but not single-threaded.

### Step 3: Dataset Point Counter

**Action**: Add atomic counter to track total dataset points examined by each kernel.

**Expected**:
- Single-threaded: ~N points examined
- Cooperative: Should be same ~N points
- If cooperative < single-threaded: Missing points!

### Step 4: Direct Instrumentation

**Action**: Add specific check for ID=74891:
- Log when it's first encountered
- Log its distance (should be 37)
- Log if it's inserted into heap
- Log if it's evicted from heap

**Expected**: Trace the life cycle of this specific neighbor in both kernels.

---

## Immediate Conclusion

The cooperative kernel has a **systematic bug** that causes it to:
1. Miss at least one correct neighbor (74891 @ distance=37)
2. Include a worse neighbor instead (5321 @ distance=38)
3. Achieve 2.3% lower precision than single-threaded

This is **NOT a minor tie-breaking issue** - the cooperative kernel is fundamentally missing correct results that the single-threaded kernel finds.

**Priority**: **CRITICAL** - Must be fixed before deployment

**Recommended Action**: Continue with Phase 2 of investigation plan - add comprehensive logging to identify the exact point where cooperative diverges from single-threaded execution.

---

**Report Generated**: 2025-11-18 23:05 UTC
**Status**: Regression confirmed, root cause investigation in progress
**Next**: Add logging infrastructure to find divergence point
