// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

// test: ccoProxy (host-proxy RDMA over the KIWI queue) put / signal / flush.
//
// FULL connectivity; kSpscQueues SPSC + kMpscQueues MPSC proxy queues polled by
// kProxyThreads proxy threads. Block u of a put kernel is a "unit": it uses
// queue u (modulo the pool size) and sends a distinct chunk to every peer:
//   recv[(src * nUnits + u) * chunk + i] = Pattern(salt, src, dst, u, i)
// Signal slot (src * kMaxUnits + u) on the receiver counts that chunk's arrival
// (put-carried or a trailing signal on the same queue, hence after the data).
// Each block waits for every sender's chunk on its unit, then the host verifies
// the receive window after a barrier.
//
// Cases:
//   put_sig_warp   put carrying SignalInc (one descriptor), ccoCoopWarp, SPSC
//   put_then_sig   put, then a separate signal() on the same queue, ccoCoopBlock
//                  (per-(queue, peer) ordering)
//   win_signal     put carrying WindowSignalAdd into a caller flag window
//   large_put      one 4 MiB put per peer on a single queue
//   kiwi_flush     put, kiwi-mode flush (checks flushDone covers the puts), then
//                  signal
//   net_flush      put, network-mode flush, overwrite the source right away,
//                  then signal: the peer must still see the original data
//   mpsc_put_sig   put_sig_warp on the MPSC pool, 4 blocks per queue pushing
//                  concurrently
//   mpsc_then_sig  put_then_sig on the MPSC pool, 4 blocks per queue
//   signal_only    every queue sends kSignalRounds SignalAdd{u+1} to one
//                  contended slot on every peer; total checked via waitSignal
//
// Launch: `mpirun -np N test_proxy_put` (one rank per MPI process, any number
// of nodes; built with MORI_WITH_MPI when WITH_MPI=ON), or `test_proxy_put [N]`
// to fork N local ranks.

#include "cco_test_harness.hpp"
#include "mori/cco/cco_proxy.hpp"

using namespace mori::cco;
using kiwi::queue::MPSCD2HQueue;
using kiwi::queue::SPSCD2HQueue;

static const size_t PER_RANK_VMM_SIZE = 512ULL * 1024 * 1024;
static constexpr int kSpscQueues = 4;
static constexpr int kMpscQueues = 2;
static constexpr int kProxyThreads = 2;
static constexpr int kMaxUnits = 8;
static constexpr size_t kChunk = 64 * 1024;
static constexpr size_t kLargeChunk = 4 * 1024 * 1024;
static constexpr int kSignalRounds = 16;
static constexpr uint32_t kClobber = 0xdeadbeefu;

enum PutMode {
  kPutSignal,          // put carrying SignalInc
  kPutThenSignal,      // put, then signal()
  kPutWindowSignal,    // put carrying WindowSignalAdd
  kPutKiwiFlush,       // put, flush<kiwi>, signal()
  kPutNetFlushClobber  // put, flush<network>, overwrite the source, signal()
};

__host__ __device__ inline uint32_t Pattern(uint32_t salt, int src, int dst, int u, size_t i) {
  return salt * 0x9E3779B1u + static_cast<uint32_t>(src) * 0x85EBCA77u +
         static_cast<uint32_t>(dst) * 0xC2B2AE3Du + static_cast<uint32_t>(u) * 0x27D4EB2Fu +
         static_cast<uint32_t>(i);
}

__host__ __device__ inline ccoGdaSignal_t Slot(int src, int unit) {
  return static_cast<ccoGdaSignal_t>(src * kMaxUnits + unit);
}

// Block u = unit u on queue u (modulo the pool). Sends one chunk to every peer,
// then waits for every sender's chunk on this unit (signal slot or flag word).
template <typename Q, typename Coop, int Mode>
__global__ void ProxyPutKernel(ccoDevComm dc, ccoWindow_t sendWin, ccoWindow_t recvWin,
                               ccoWindow_t flagWin, uint32_t* sendLocal, uint64_t* flagLocal,
                               size_t chunk, int nUnits, int* err) {
  const int u = blockIdx.x;
  const int me = dc.rank;
  const int n = dc.worldSize;
  ccoProxy<Q> proxy{dc, u};
  auto srcOff = [&](int p) { return (static_cast<size_t>(p) * nUnits + u) * chunk; };
  const size_t dstOff = (static_cast<size_t>(me) * nUnits + u) * chunk;
  for (int p = 0; p < n; p++) {
    if (p == me) continue;
    if constexpr (Mode == kPutSignal) {
      proxy.put(p, recvWin, dstOff, sendWin, srcOff(p), chunk, ccoGda_SignalInc{Slot(me, u)},
                Coop{});
    } else if constexpr (Mode == kPutWindowSignal) {
      proxy.put(p, recvWin, dstOff, sendWin, srcOff(p), chunk,
                ccoGda_WindowSignalAdd{flagWin, Slot(me, u) * sizeof(uint64_t), 1}, Coop{});
    } else {
      proxy.put(p, recvWin, dstOff, sendWin, srcOff(p), chunk, ccoGda_NoSignal{}, Coop{});
      if constexpr (Mode == kPutThenSignal) {
        proxy.signal(p, ccoGda_SignalAdd{Slot(me, u), 1}, Coop{});
      }
    }
  }
  if constexpr (Mode == kPutKiwiFlush) {
    proxy.template flush<ccoProxyFlushKiwi>(Coop{});
    if constexpr (ccoProxy<Q>::kSpsc) {
      // flushDone covers the FLUSH, which sits after every put of this queue.
      if (threadIdx.x == 0 &&
          __hip_atomic_load(proxy.flushDone, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) <
              proxy.queue.producer_index) {
        printf("[rank %d] kiwi_flush: unit %d flushDone behind the queue\n", me, u);
        atomicExch(err, 1);
      }
    }
  } else if constexpr (Mode == kPutNetFlushClobber) {
    proxy.template flush<ccoProxyFlushNetwork>(Coop{});
    // Every put of this unit completed: its sources may be reused right away.
    for (int p = 0; p < n; p++) {
      if (p == me) continue;
      uint32_t* src = sendLocal + srcOff(p) / sizeof(uint32_t);
      for (size_t i = threadIdx.x; i < chunk / sizeof(uint32_t); i += blockDim.x) src[i] = kClobber;
    }
    __threadfence_system();
    __syncthreads();
  }
  if constexpr (Mode == kPutKiwiFlush || Mode == kPutNetFlushClobber) {
    for (int p = 0; p < n; p++) {
      if (p != me) proxy.signal(p, ccoGda_SignalInc{Slot(me, u)}, Coop{});
    }
  }
  if (threadIdx.x == 0) {
    for (int s = 0; s < n; s++) {
      if (s == me) continue;
      if constexpr (Mode == kPutWindowSignal) {
        while (__hip_atomic_load(&flagLocal[Slot(s, u)], __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_SYSTEM) < 1) {
        }
      } else {
        proxy.waitSignal(Slot(s, u), 1);
      }
    }
  }
}

// Every SPSC queue sends kSignalRounds SignalAdd{u+1} to `slot` on every peer,
// then block 0 consumes the expected total and checks nothing is left over.
__global__ void ProxySignalKernel(ccoDevComm dc, ccoGdaSignal_t slot, int* err) {
  const int u = blockIdx.x;
  const int me = dc.rank;
  const int n = dc.worldSize;
  ccoProxy<> proxy{dc, u};
  for (int k = 0; k < kSignalRounds; k++) {
    for (int p = 0; p < n; p++) {
      if (p != me) proxy.signal(p, ccoGda_SignalAdd{slot, static_cast<uint64_t>(u + 1)});
    }
  }
  if (u == 0 && threadIdx.x == 0) {
    const uint64_t perPeer = static_cast<uint64_t>(kSignalRounds) * gridDim.x * (gridDim.x + 1) / 2;
    proxy.waitSignal(slot, perPeer * (n - 1));
    const uint64_t rest = proxy.readSignal(slot);
    if (rest != 0) {
      printf("[rank %d] signal_only: %llu left after consuming the expected total\n", me,
             (unsigned long long)rest);
      atomicExch(err, 1);
    }
  }
}

__global__ void ProxyResetKernel(ccoDevComm dc, int nSlots) {
  ccoProxy<> proxy{dc, 0};  // no push: reads the slots' signal state only
  for (int i = threadIdx.x; i < nSlots; i += blockDim.x) {
    proxy.resetSignal(static_cast<ccoGdaSignal_t>(i));
  }
}

struct Ctx {
  ccoComm* comm;
  ccoDevComm dc;
  hipStream_t stream;
  int rank;
  int nranks;
  ccoWindow_t sendWin, recvWin, flagWin;
  void *sendBuf, *recvBuf, *flagBuf;
  size_t bufSize;
  int nSlots;
  int* dErr;

  void Sync() {
    HIP_CHECK(hipStreamSynchronize(stream));
    ccoBarrierAll(comm);
  }

  // Reset signal slots, flags, recv window; fill send window for this salt.
  void Fresh(uint32_t salt, size_t chunk, int nUnits) {
    ProxyResetKernel<<<1, 256, 0, stream>>>(dc, nSlots);
    HIP_CHECK(hipMemsetAsync(flagBuf, 0, sizeof(uint64_t) * nSlots, stream));
    HIP_CHECK(hipMemsetAsync(recvBuf, 0xff, bufSize, stream));
    std::vector<uint32_t> host(bufSize / sizeof(uint32_t), 0);
    for (int p = 0; p < nranks; p++) {
      for (int u = 0; u < nUnits; u++) {
        uint32_t* dst =
            host.data() + (static_cast<size_t>(p) * nUnits + u) * chunk / sizeof(uint32_t);
        for (size_t i = 0; i < chunk / sizeof(uint32_t); i++) dst[i] = Pattern(salt, rank, p, u, i);
      }
    }
    HIP_CHECK(hipMemcpyAsync(sendBuf, host.data(), bufSize, hipMemcpyHostToDevice, stream));
    Sync();
  }

  int Verify(const char* name, uint32_t salt, size_t chunk, int nUnits) {
    std::vector<uint32_t> host(bufSize / sizeof(uint32_t));
    HIP_CHECK(hipMemcpy(host.data(), recvBuf, bufSize, hipMemcpyDeviceToHost));
    int bad = 0;
    for (int s = 0; s < nranks && !bad; s++) {
      if (s == rank) continue;
      for (int u = 0; u < nUnits && !bad; u++) {
        const uint32_t* got =
            host.data() + (static_cast<size_t>(s) * nUnits + u) * chunk / sizeof(uint32_t);
        for (size_t i = 0; i < chunk / sizeof(uint32_t); i++) {
          const uint32_t want = Pattern(salt, s, rank, u, i);
          if (got[i] != want) {
            fprintf(stderr, "[rank %d] %s: src=%d unit=%d word %zu: got %08x want %08x\n", rank,
                    name, s, u, i, got[i], want);
            bad = 1;
            break;
          }
        }
      }
    }
    return Harvest(name, bad);
  }

  int Harvest(const char* name, int hostBad) {
    int dBad = 0;
    HIP_CHECK(hipMemcpy(&dBad, dErr, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemset(dErr, 0, sizeof(int)));
    const int bad = (hostBad || dBad) ? 1 : 0;
    printf("[rank %d]   [%-13s] %s\n", rank, name, bad ? "FAIL" : "PASS");
    ccoBarrierAll(comm);
    return bad;
  }
};

template <typename Q, typename Coop, int Mode>
static int RunPutCase(Ctx& c, const char* name, uint32_t salt, size_t chunk, int nUnits,
                      int blockDim) {
  c.Fresh(salt, chunk, nUnits);
  ProxyPutKernel<Q, Coop, Mode><<<nUnits, blockDim, 0, c.stream>>>(
      c.dc, c.sendWin, c.recvWin, c.flagWin, static_cast<uint32_t*>(c.sendBuf),
      static_cast<uint64_t*>(c.flagBuf), chunk, nUnits, c.dErr);
  c.Sync();
  return c.Verify(name, salt, chunk, nUnits);
}

static int RunSignalCase(Ctx& c) {
  c.Fresh(/*salt=*/0, kChunk, kSpscQueues);
  const ccoGdaSignal_t slot = static_cast<ccoGdaSignal_t>(c.nSlots - 1);
  ProxySignalKernel<<<kSpscQueues, 64, 0, c.stream>>>(c.dc, slot, c.dErr);
  c.Sync();
  return c.Harvest("signal_only", 0);
}

int run_test(int rank, int nranks, const ccoUniqueId& uid) {
  g_rank = rank;
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  const int dev = rank % numDevices;
  HIP_CHECK(hipSetDevice(dev));
  printf("[rank %d/%d] pid=%d GPU=%d\n", rank, nranks, getpid(), dev);

  Ctx c{};
  c.rank = rank;
  c.nranks = nranks;
  if (ccoCommCreate(uid, nranks, rank, PER_RANK_VMM_SIZE, &c.comm) != 0) {
    fprintf(stderr, "[rank %d] CommCreate failed\n", rank);
    return 1;
  }

  // One buffer size covers every chunk layout used below.
  c.bufSize = std::max(nranks * kMaxUnits * kChunk, nranks * kLargeChunk);
  c.nSlots = nranks * kMaxUnits + 1;  // per (sender, unit) + one contended slot
  if (ccoWindowRegister(c.comm, c.bufSize, &c.sendWin, &c.sendBuf) != 0 ||
      ccoWindowRegister(c.comm, c.bufSize, &c.recvWin, &c.recvBuf) != 0 ||
      ccoWindowRegister(c.comm, sizeof(uint64_t) * c.nSlots, &c.flagWin, &c.flagBuf) != 0) {
    fprintf(stderr, "[rank %d] WindowRegister failed\n", rank);
    return 1;
  }

  ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = CCO_GDA_CONNECTION_FULL;
  reqs.gdaContextCount = 1;
  reqs.gdaSignalCount = c.nSlots;
  reqs.gdaCounterCount = 0;
  reqs.proxySpscQueueCount = kSpscQueues;
  reqs.proxyMpscQueueCount = kMpscQueues;
  reqs.proxyThreadCount = kProxyThreads;
  if (ccoDevCommCreate(c.comm, &reqs, &c.dc) != 0) {
    fprintf(stderr, "[rank %d] DevCommCreate failed\n", rank);
    return 1;
  }
  if (c.dc.gdaConnType == CCO_GDA_CONNECTION_NONE || c.dc.proxy.numSpscQueues != kSpscQueues ||
      c.dc.proxy.numMpscQueues != kMpscQueues) {
    fprintf(stderr, "[rank %d] proxy queues missing (gdaConnType=%d spsc=%d mpsc=%d)\n", rank,
            (int)c.dc.gdaConnType, c.dc.proxy.numSpscQueues, c.dc.proxy.numMpscQueues);
    return 1;
  }
  HIP_CHECK(hipMalloc(&c.dErr, sizeof(int)));
  HIP_CHECK(hipMemset(c.dErr, 0, sizeof(int)));
  HIP_CHECK(hipStreamCreate(&c.stream));
  ccoBarrierAll(c.comm);

  int fails = 0;
  fails += RunPutCase<SPSCD2HQueue, ccoCoopWarp, kPutSignal>(c, "put_sig_warp", 1, kChunk,
                                                             kSpscQueues, 64);
  fails += RunPutCase<SPSCD2HQueue, ccoCoopBlock, kPutThenSignal>(c, "put_then_sig", 2, kChunk,
                                                                  kSpscQueues, 256);
  fails += RunPutCase<SPSCD2HQueue, ccoCoopWarp, kPutWindowSignal>(c, "win_signal", 3, kChunk,
                                                                   kSpscQueues, 64);
  fails += RunPutCase<SPSCD2HQueue, ccoCoopWarp, kPutSignal>(c, "large_put", 4, kLargeChunk, 1, 64);
  fails += RunPutCase<SPSCD2HQueue, ccoCoopWarp, kPutKiwiFlush>(c, "kiwi_flush", 5, kChunk,
                                                                kSpscQueues, 64);
  fails += RunPutCase<SPSCD2HQueue, ccoCoopBlock, kPutNetFlushClobber>(c, "net_flush", 6, kChunk,
                                                                       kSpscQueues, 256);
  // kMaxUnits blocks over kMpscQueues MPSC queues: several concurrent producers per queue.
  fails += RunPutCase<MPSCD2HQueue, ccoCoopWarp, kPutSignal>(c, "mpsc_put_sig", 7, kChunk,
                                                             kMaxUnits, 64);
  fails += RunPutCase<MPSCD2HQueue, ccoCoopBlock, kPutThenSignal>(c, "mpsc_then_sig", 8, kChunk,
                                                                  kMaxUnits, 256);
  fails += RunSignalCase(c);

  HIP_CHECK(hipStreamDestroy(c.stream));
  HIP_CHECK(hipFree(c.dErr));
  ccoDevCommDestroy(c.comm, &c.dc);
  ccoWindowDeregister(c.comm, c.flagWin);
  ccoWindowDeregister(c.comm, c.recvWin);
  ccoWindowDeregister(c.comm, c.sendWin);
  ccoMemFree(c.comm, c.flagBuf);
  ccoMemFree(c.comm, c.recvBuf);
  ccoMemFree(c.comm, c.sendBuf);
  ccoCommDestroy(c.comm);

  printf("[rank %d] %s\n", rank, fails == 0 ? "PASSED" : "FAILED");
  return fails == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
  const int rc = ccoTestMain(argc, argv, "CCO proxy put", "/tmp/cco_proxy_put_uid", 19890);
#ifdef MORI_WITH_MPI
  int initialized = 0, finalized = 0;
  MPI_Initialized(&initialized);
  MPI_Finalized(&finalized);
  if (initialized && !finalized) MPI_Finalize();
#endif
  return rc;
}
