// SM90 capability scaffold.  This intentionally contains no WGMMA/TMA
// implementation; the Python adapter keeps the artifact metadata-only until
// a Hopper device validates the real CUTLASS path.
#include <cuda_runtime_api.h>

extern "C" __global__ void xqt_sm90_wgmma_scaffold() {}

extern "C" int xqt_sm90_wgmma_scaffold_version() { return 1; }
