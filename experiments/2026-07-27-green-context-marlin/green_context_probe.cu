// Green Context feasibility probe for GH200 (CUDA 13, sm_90a).
// Track A, plan §4.1-4.2.
//
// Validates the CUDA green-context mechanism before any vLLM/Marlin work:
//   1. Query the device SM resource (smCount, minSmPartitionSize, smCoscheduledAlignment).
//   2. Split SMs into a small "cold" partition + a large "hot" remainder.
//   3. Create two green contexts and a stream in each (driver API).
//   4. Verify stream -> green-context mapping and the disjoint SM counts.
//   5. Launch a busy kernel on each stream: solo, then concurrent.
//   6. Report whether the two green contexts actually execute concurrently
//      (union << serial sum) — the binary gate for all of Track A.
//
// Build:  nvcc -O2 -arch=sm_90a -o green_context_probe green_context_probe.cu
// Run:    CUDA_DEVICE_MAX_CONNECTIONS=8 ./green_context_probe [iters]
//
// If <<<>>> launches on a green-context stream error on this driver, the probe
// prints the error clearly — that itself is a finding (means we need
// per-thread cuCtxFromGreenCtx+cuCtxSetCurrent or driver-API cuLaunchKernel).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CK(x) do { CUresult _r = (x); if (_r != CUDA_SUCCESS) { \
    const char* e = nullptr; cuGetErrorString(_r, &e); \
    fprintf(stderr, "DRIVER FAIL %s = %d (%s) @ line %d\n", #x, (int)_r, e ? e : "?", __LINE__); \
    return 1; } } while (0)
#define CKRT(x) do { cudaError_t _r = (x); if (_r != cudaSuccess) { \
    fprintf(stderr, "RT FAIL %s = %s @ line %d\n", #x, cudaGetErrorString(_r), __LINE__); \
    return 1; } } while (0)

__global__ void cold_busy(long iters, int* out) {
    int bid = blockIdx.x;
    volatile long s = 0;
    for (long i = 0; i < iters; ++i) s += i;
    if (threadIdx.x == 0) out[bid] = (int)s;
}
__global__ void hot_busy(long iters, int* out) {
    int bid = blockIdx.x;
    volatile long s = 0;
    for (long i = 0; i < iters; ++i) s += i;
    if (threadIdx.x == 0) out[bid] = (int)s;
}

static float time_kernel(cudaStream_t s, void (*k)(long, int*),
                         int grid, long iters, int* out) {
    cudaEvent_t a, b;
    CKRT(cudaEventCreate(&a)); CKRT(cudaEventCreate(&b));
    CKRT(cudaEventRecord(a, s));
    k<<<grid, 1, 0, s>>>(iters, out);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) {
        fprintf(stderr, "LAUNCH FAIL on stream %p grid=%d: %s\n", (void*)s, grid, cudaGetErrorString(le));
        CKRT(cudaEventDestroy(a)); CKRT(cudaEventDestroy(b));
        return -1.0f;
    }
    CKRT(cudaEventRecord(b, s));
    CKRT(cudaEventSynchronize(b));
    float ms = 0.0f;
    CKRT(cudaEventElapsedTime(&ms, a, b));
    CKRT(cudaEventDestroy(a)); CKRT(cudaEventDestroy(b));
    return ms;
}

int main(int argc, char** argv) {
    long iters = (argc > 1) ? atol(argv[1]) : 20000000L;  // ~ms-scale busy loop

    printf("=== Green Context feasibility probe (GH200 / CUDA %d) ===\n", CUDA_VERSION);
    int max_conn = -1;
    cudaDeviceGetAttribute(&max_conn, cudaDevAttrMultiProcessorCount, 0);
    // Note: CUDA_DEVICE_MAX_CONNECTIONS is an env var; echo it.
    printf("env CUDA_DEVICE_MAX_CONNECTIONS=%s\n", getenv("CUDA_DEVICE_MAX_CONNECTIONS") ? getenv("CUDA_DEVICE_MAX_CONNECTIONS") : "<unset>");

    // --- 1. init driver + primary context (runtime) ---
    CK(cuInit(0));
    CKRT(cudaSetDevice(0));
    CKRT(cudaFree(0));  // force primary context active
    CUdevice dev = 0;

    // --- 2. query full SM resource ---
    CUdevResource full{};
    CK(cuDeviceGetDevResource(dev, &full, CU_DEV_RESOURCE_TYPE_SM));
    printf("device SM: smCount=%u minSmPartitionSize=%u smCoscheduledAlignment=%u\n",
           full.sm.smCount, full.sm.minSmPartitionSize, full.sm.smCoscheduledAlignment);

    // --- 3. split: one small cold group (>=16 SMs) + remainder (hot) ---
    // Hopper alignment is typically 8; minCount=16 requests a 16-SM cold partition.
    CUdevResource coldRes{};
    CUdevResource remain{};
    unsigned int nbGroups = 1;
    CK(cuDevSmResourceSplitByCount(&coldRes, &nbGroups, &full, &remain, /*useFlags*/ 0, /*minCount*/ 16));
    printf("split: nbGroups=%u  cold.smCount=%u  remain.smCount=%u  (sum=%u, device=%u)\n",
           nbGroups, coldRes.sm.smCount, remain.sm.smCount,
           coldRes.sm.smCount + remain.sm.smCount, full.sm.smCount);
    if (coldRes.sm.smCount + remain.sm.smCount > full.sm.smCount) {
        fprintf(stderr, "WARN: partition sum exceeds device SMs — not disjoint?\n");
    }

    // --- 4. descriptors + green contexts + streams ---
    CUdevResourceDesc coldDesc{}, hotDesc{};
    CK(cuDevResourceGenerateDesc(&coldDesc, &coldRes, 1));
    CK(cuDevResourceGenerateDesc(&hotDesc, &remain, 1));

    CUgreenCtx coldCtx, hotCtx;
    CK(cuGreenCtxCreate(&coldCtx, coldDesc, dev, CU_GREEN_CTX_DEFAULT_STREAM));
    CK(cuGreenCtxCreate(&hotCtx,  hotDesc,  dev, CU_GREEN_CTX_DEFAULT_STREAM));

    unsigned long long cid = 0, hid = 0;
    CK(cuGreenCtxGetId(coldCtx, &cid));
    CK(cuGreenCtxGetId(hotCtx,  &hid));
    printf("greenCtxIds: cold=%llu hot=%llu (distinct=%d)\n", cid, hid, cid != hid);

    CUstream coldStream, hotStream;
    // CU_STREAM_NON_BLOCKING is REQUIRED for cuGreenCtxStreamCreate (flags=0 -> INVALID_VALUE).
    // Query the valid priority range and use the greatest (highest-priority) value.
    int leastPri = 0, greatestPri = 0;
    cudaDeviceGetStreamPriorityRange(&leastPri, &greatestPri);
    printf("stream priority range: least=%d greatest=%d (using greatest)\n", leastPri, greatestPri);
    CK(cuGreenCtxStreamCreate(&coldStream, coldCtx, CU_STREAM_NON_BLOCKING, greatestPri));
    CK(cuGreenCtxStreamCreate(&hotStream,  hotCtx,  CU_STREAM_NON_BLOCKING, greatestPri));

    // verify stream -> ctx mapping
    CUgreenCtx g0 = nullptr, g1 = nullptr;
    CK(cuStreamGetGreenCtx(coldStream, &g0));
    CK(cuStreamGetGreenCtx(hotStream,  &g1));
    printf("stream->ctx verify: cold matches=%d hot matches=%d\n", g0 == coldCtx, g1 == hotCtx);

    // also query each green ctx's *actual* provisioned resource (disjoint proof)
    CUdevResource coldGot{}, hotGot{};
    CK(cuGreenCtxGetDevResource(coldCtx, &coldGot, CU_DEV_RESOURCE_TYPE_SM));
    CK(cuGreenCtxGetDevResource(hotCtx,  &hotGot,  CU_DEV_RESOURCE_TYPE_SM));
    printf("provisioned: cold.smCount=%u hot.smCount=%u (sum=%u)\n",
           coldGot.sm.smCount, hotGot.sm.smCount, coldGot.sm.smCount + hotGot.sm.smCount);

    // --- 5. buffers ---
    int *d_cold = nullptr, *d_hot = nullptr, *h_pin = nullptr;
    CKRT(cudaMalloc(&d_cold, 4096));
    CKRT(cudaMalloc(&d_hot, 4096));
    CKRT(cudaHostAlloc(&h_pin, 4096, cudaHostAllocDefault));  // pinned host (UVA) probe
    // confirm pinned ptr is UVA-readable from device
    cudaPointerAttributes pa{};
    cudaPointerGetAttributes(&pa, h_pin);
    printf("pinned host ptr: type=%d deviceAccessible=%d\n", (int)pa.type, pa.devicePointer != nullptr);

    // --- 6. timing: solo then concurrent ---
    int coldGrid = coldGot.sm.smCount;
    int hotGrid  = hotGot.sm.smCount;
    printf("grids: cold=%d hot=%d iters=%ld\n", coldGrid, hotGrid, iters);

    float t_cold_solo = time_kernel((cudaStream_t)coldStream, cold_busy, coldGrid, iters, d_cold);
    float t_hot_solo  = time_kernel((cudaStream_t)hotStream,  hot_busy,  hotGrid,  iters, d_hot);
    printf("SOLO ms: cold=%.3f hot=%.3f\n", t_cold_solo, t_hot_solo);
    if (t_cold_solo < 0 || t_hot_solo < 0) {
        fprintf(stderr, "Solo launch failed — see LAUNCH FAIL above. This is a finding: "
                "<<<>>> on a green-context stream is not supported on this driver; "
                "need per-thread cuCtxFromGreenCtx or driver-API cuLaunchKernel.\n");
        return 2;
    }

    // concurrent: record events on both, overlap cold+hot, sync both
    cudaEvent_t c0, c1, h0, h1;
    CKRT(cudaEventCreate(&c0)); CKRT(cudaEventCreate(&c1));
    CKRT(cudaEventCreate(&h0)); CKRT(cudaEventCreate(&h1));
    CKRT(cudaEventRecord(c0, (cudaStream_t)coldStream));
    cold_busy<<<coldGrid, 1, 0, (cudaStream_t)coldStream>>>(iters, d_cold);
    CKRT(cudaEventRecord(c1, (cudaStream_t)coldStream));
    CKRT(cudaEventRecord(h0, (cudaStream_t)hotStream));
    hot_busy<<<hotGrid, 1, 0, (cudaStream_t)hotStream>>>(iters, d_hot);
    CKRT(cudaEventRecord(h1, (cudaStream_t)hotStream));
    CKRT(cudaEventSynchronize(c1));
    CKRT(cudaEventSynchronize(h1));
    float t_cold_ov = 0, t_hot_ov = 0;
    CKRT(cudaEventElapsedTime(&t_cold_ov, c0, c1));
    CKRT(cudaEventElapsedTime(&t_hot_ov,  h0, h1));
    float union_ms = (t_cold_ov > t_hot_ov) ? t_cold_ov : t_hot_ov;
    float serial_ms = t_cold_solo + t_hot_solo;
    printf("CONCURRENT ms: cold_ov=%.3f hot_ov=%.3f  union=%.3f  serial_sum=%.3f\n",
           t_cold_ov, t_hot_ov, union_ms, serial_ms);
    printf("CONCURRENCY: overlap_speedup=%.2fx  (1.0=no overlap, ~2.0=full overlap of 2 equal kernels)\n",
           serial_ms / union_ms);

    // --- 7. correctness sanity: outputs are non-zero ---
    int hc = 0, hh = 0;
    CKRT(cudaMemcpy(&hc, d_cold, sizeof(int), cudaMemcpyDeviceToHost));
    CKRT(cudaMemcpy(&hh, d_hot,  sizeof(int), cudaMemcpyDeviceToHost));
    printf("outputs: cold_out[0]=%d hot_out[0]=%d (nonzero=%d)\n", hc, hh, hc != 0 && hh != 0);

    printf("=== probe done ===\n");
    return 0;
}
