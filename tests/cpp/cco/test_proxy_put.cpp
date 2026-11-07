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
// queue u (MPSC: modulo the pool size) and sends a distinct chunk to every peer:
//   recv[(src * nUnits + u) * chunk + i] = Pattern(salt, src, dst, u, i)
// Signal slot (src * kMaxUnits + u) on the receiver counts that chunk's arrival
// (put-carried or a trailing signal on the same queue, hence after the data).
// Each block waits for every sender's chunk on its unit, then the host verifies
// the receive window after a barrier. The host fills the send window, except
// in the kernel_payload cases.
//
// Cases:
//   put_sig_warp   put carrying SignalInc (one descriptor), ccoCoopWarp, SPSC
//   put_then_sig   put, then a separate signal() on the same queue, ccoCoopBlock
//                  (per-(queue, peer) ordering)
//   win_signal     put carrying WindowSignalAdd into a caller flag window
//   large_put      one 4 MiB put per peer on a single queue
//   flush_count    put, flush (checks it advanced flushDone by exactly one),
//                  then signal
//   flush_reuse    put, flush, overwrite the source right away, then signal:
//                  the peer must still see the original data
//   kernel_payload every wavefront of a 4-wavefront block writes part of the
//                  chunk, then the block puts it (ccoCoopBlock): the put's
//                  fence must publish stores of the non-issuing wavefronts
//   sq_full        one queue puts its chunk kSqRepeats times per peer with no
//                  flush in between, more WRs than a proxy QP's SQ holds, so
//                  the proxy waits for SQ credit
//   mpsc_put_sig   put_sig_warp on the MPSC pool, 4 blocks per queue pushing
//                  concurrently
//   mpsc_then_sig  put_then_sig on the MPSC pool, 4 blocks per queue
//   mpsc_flush     flush_reuse on the MPSC pool, 4 blocks per queue flushing
//                  concurrently
//   mpsc_payload   kernel_payload on the MPSC pool
//   mpsc_stress    kStressBlocks x kStressWarps wavefronts on one MPSC queue,
//                  each doing kStressRounds rounds of put, flushing after most
//                  rounds. After each flush a wavefront overwrites the sources
//                  of every put since its previous flush. Ticket fetch-adds and
//                  FLUSH claims are separate atomics, so under this contention
//                  FLUSHes can land in the ring out of ticket order.
//   signal_only    every queue sends kSignalRounds SignalAdd{u+1} to one
//                  contended slot on every peer; total checked via waitSignal
//   small_spsc,    on a second DevComm whose rings hold kSmallCapacity lines:
//   small_mpsc     each unit sends its chunk as kPieces puts per peer (the
//                  last carrying the signal), so every ring wraps many times
//                  and producers wait for ring space (SPSC: 1 producer per
//                  queue; MPSC: 4)
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
static constexpr int kStressBlocks = 8;
static constexpr int kStressWarps = 4;  // wavefronts per block
static constexpr int kStressRounds = 8;
static constexpr size_t kStressChunk = 16 * 1024;
static constexpr int kSmallCapacity = 16;  // ring lines: a put + signal takes 2
static constexpr int kPieces = 64;         // puts per chunk in the small-ring cases
static constexpr int kSqRepeats = 2048;    // puts per peer in sq_full (SQ depth: 512)
static constexpr uint32_t kClobber = 0xdeadbeefu;

enum PutMode {
  kPutSignal,         // put carrying SignalInc
  kPutThenSignal,     // put, then signal()
  kPutWindowSignal,   // put carrying WindowSignalAdd
  kPutFlushCount,     // put, flush (flushDone advances by one), signal()
  kPutFlushClobber,   // put, flush, overwrite the source, signal()
  kPutKernelPayload,  // the block writes the source, then put carrying SignalInc
  kPutRepeat,         // kSqRepeats puts of the same chunk, the last carrying SignalInc
  kPutPieces          // kPieces puts per chunk, the last carrying SignalInc
};

__host__ __device__ inline uint32_t Pattern(uint32_t salt, int src, int dst, int u, size_t i) {
  return salt * 0x9E3779B1u + static_cast<uint32_t>(src) * 0x85EBCA77u +
         static_cast<uint32_t>(dst) * 0xC2B2AE3Du + static_cast<uint32_t>(u) * 0x27D4EB2Fu +
         static_cast<uint32_t>(i);
}

__host__ __device__ inline ccoGdaSignal_t Slot(int src, int unit) {
  return static_cast<ccoGdaSignal_t>(src * kMaxUnits + unit);
}

// Block u = unit u on queue u (MPSC: modulo the pool). Sends one chunk to every peer,
// then waits for every sender's chunk on this unit (signal slot or flag word).
template <typename Q, typename Coop, int Mode>
__global__ void ProxyPutKernel(ccoDevComm dc, ccoWindow_t sendWin, ccoWindow_t recvWin,
                               ccoWindow_t flagWin, uint32_t* sendLocal, uint64_t* flagLocal,
                               uint32_t salt, size_t chunk, int nUnits, int* err) {
  const int u = blockIdx.x;
  const int me = dc.rank;
  const int n = dc.worldSize;
  ccoProxy<Q> proxy{dc, u};
  auto srcOff = [&](int p) { return (static_cast<size_t>(p) * nUnits + u) * chunk; };
  const size_t dstOff = (static_cast<size_t>(me) * nUnits + u) * chunk;
  if constexpr (Mode == kPutKernelPayload) {
    // Every thread of the block writes the chunks; put() publishes them.
    for (int p = 0; p < n; p++) {
      if (p == me) continue;
      uint32_t* src = sendLocal + srcOff(p) / sizeof(uint32_t);
      for (size_t i = threadIdx.x; i < chunk / sizeof(uint32_t); i += blockDim.x) {
        src[i] = Pattern(salt, me, p, u, i);
      }
    }
  }
  for (int p = 0; p < n; p++) {
    if (p == me) continue;
    if constexpr (Mode == kPutSignal || Mode == kPutKernelPayload) {
      proxy.put(p, recvWin, dstOff, sendWin, srcOff(p), chunk, ccoGda_SignalInc{Slot(me, u)},
                Coop{});
    } else if constexpr (Mode == kPutWindowSignal) {
      proxy.put(p, recvWin, dstOff, sendWin, srcOff(p), chunk,
                ccoGda_WindowSignalAdd{flagWin, Slot(me, u) * sizeof(uint64_t), 1}, Coop{});
    } else if constexpr (Mode == kPutRepeat) {
      for (int k = 0; k < kSqRepeats - 1; k++) {
        proxy.put(p, recvWin, dstOff, sendWin, srcOff(p), chunk, ccoGda_NoSignal{}, Coop{});
      }
      proxy.put(p, recvWin, dstOff, sendWin, srcOff(p), chunk, ccoGda_SignalInc{Slot(me, u)},
                Coop{});
    } else if constexpr (Mode == kPutPieces) {
      const size_t piece = chunk / kPieces;
      for (int k = 0; k < kPieces - 1; k++) {
        proxy.put(p, recvWin, dstOff + k * piece, sendWin, srcOff(p) + k * piece, piece,
                  ccoGda_NoSignal{}, Coop{});
      }
      const size_t last = (kPieces - 1) * piece;
      proxy.put(p, recvWin, dstOff + last, sendWin, srcOff(p) + last, chunk - last,
                ccoGda_SignalInc{Slot(me, u)}, Coop{});
    } else {
      proxy.put(p, recvWin, dstOff, sendWin, srcOff(p), chunk, ccoGda_NoSignal{}, Coop{});
      if constexpr (Mode == kPutThenSignal) {
        proxy.signal(p, ccoGda_SignalAdd{Slot(me, u), 1}, Coop{});
      }
    }
  }
  if constexpr (Mode == kPutFlushCount) {
    uint64_t before = 0;
    if constexpr (ccoProxy<Q>::kSpsc) {
      before = __hip_atomic_load(proxy.flushDone, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    proxy.flush(Coop{});
    if constexpr (ccoProxy<Q>::kSpsc) {
      // This wavefront owns the queue: its FLUSH is the only one in flight.
      const uint64_t after =
          __hip_atomic_load(proxy.flushDone, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      if (threadIdx.x == 0 && after != before + 1) {
        printf("[rank %d] flush_count: unit %d flushDone %llu -> %llu\n", me, u,
               (unsigned long long)before, (unsigned long long)after);
        atomicExch(err, 1);
      }
    }
  } else if constexpr (Mode == kPutFlushClobber) {
    proxy.flush(Coop{});
    // Every put of this unit completed: its sources may be reused right away.
    for (int p = 0; p < n; p++) {
      if (p == me) continue;
      uint32_t* src = sendLocal + srcOff(p) / sizeof(uint32_t);
      for (size_t i = threadIdx.x; i < chunk / sizeof(uint32_t); i += blockDim.x) src[i] = kClobber;
    }
    __threadfence_system();
    __syncthreads();
  }
  if constexpr (Mode == kPutFlushCount || Mode == kPutFlushClobber) {
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

// Every wavefront w (of kStressBlocks x kStressWarps) shares MPSC queue 0. In
// round r it puts unit u = w * kStressRounds + r to every peer, then flushes in
// the last round and in two of every three others. A flush covers every
// earlier put of the wavefront, so it then overwrites the sources of all units
// since its previous flush.
// Finally each wavefront adds 1 to `slot` on every peer, and block 0 waits for
// every peer's wavefronts.
__global__ void ProxyStressKernel(ccoDevComm dc, ccoWindow_t sendWin, ccoWindow_t recvWin,
                                  uint32_t* sendLocal, ccoGdaSignal_t slot, int* err) {
  const int lane = __lane_id();
  const int waveSize = __builtin_amdgcn_wavefrontsize();
  const int w = blockIdx.x * (blockDim.x / waveSize) + threadIdx.x / waveSize;
  const int nWaves = gridDim.x * (blockDim.x / waveSize);
  const int nUnits = nWaves * kStressRounds;
  const int me = dc.rank;
  const int n = dc.worldSize;
  ccoProxy<MPSCD2HQueue> proxy{dc, 0};
  auto off = [&](int rank, int u) {
    return (static_cast<size_t>(rank) * nUnits + u) * kStressChunk;
  };
  int clean = 0;  // first round whose sources are not yet overwritten
  for (int r = 0; r < kStressRounds; r++) {
    const int u = w * kStressRounds + r;
    for (int p = 0; p < n; p++) {
      if (p != me) {
        proxy.put(p, recvWin, off(me, u), sendWin, off(p, u), kStressChunk, ccoGda_NoSignal{},
                  ccoCoopWarp{});
      }
    }
    if (r == kStressRounds - 1 || (w + r) % 3 != 2) {
      proxy.flush(ccoCoopWarp{});
      for (int k = w * kStressRounds + clean; k <= u; k++) {
        for (int p = 0; p < n; p++) {
          if (p == me) continue;
          uint32_t* src = sendLocal + off(p, k) / sizeof(uint32_t);
          for (size_t i = lane; i < kStressChunk / sizeof(uint32_t); i += waveSize)
            src[i] = kClobber;
        }
      }
      __threadfence_system();
      clean = r + 1;
    }
  }
  for (int p = 0; p < n; p++) {
    if (p != me) proxy.signal(p, ccoGda_SignalAdd{slot, 1}, ccoCoopWarp{});
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    proxy.waitSignal(slot, static_cast<uint64_t>(nWaves) * (n - 1));
    if (proxy.readSignal(slot) != 0) {
      printf("[rank %d] mpsc_stress: extra signals\n", me);
      atomicExch(err, 1);
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
  ccoDevComm dc;       // default rings
  ccoDevComm dcSmall;  // kSmallCapacity-line rings
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

  // Reset d's signal slots, flags, recv window; fill the send window for this
  // salt (or clobber it, when the kernel writes the sources itself).
  void Fresh(const ccoDevComm& d, uint32_t salt, size_t chunk, int nUnits, bool fillSend = true) {
    ProxyResetKernel<<<1, 256, 0, stream>>>(d, nSlots);
    HIP_CHECK(hipMemsetAsync(flagBuf, 0, sizeof(uint64_t) * nSlots, stream));
    HIP_CHECK(hipMemsetAsync(recvBuf, 0xff, bufSize, stream));
    if (!fillSend) {
      HIP_CHECK(hipMemsetAsync(sendBuf, 0x5a, bufSize, stream));
      Sync();
      return;
    }
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
                      int blockDim, bool smallRings = false) {
  const ccoDevComm& d = smallRings ? c.dcSmall : c.dc;
  c.Fresh(d, salt, chunk, nUnits, /*fillSend=*/Mode != kPutKernelPayload);
  ProxyPutKernel<Q, Coop, Mode><<<nUnits, blockDim, 0, c.stream>>>(
      d, c.sendWin, c.recvWin, c.flagWin, static_cast<uint32_t*>(c.sendBuf),
      static_cast<uint64_t*>(c.flagBuf), salt, chunk, nUnits, c.dErr);
  c.Sync();
  return c.Verify(name, salt, chunk, nUnits);
}

static int RunStressCase(Ctx& c) {
  const int nWaves = kStressBlocks * kStressWarps;
  const int nUnits = nWaves * kStressRounds;
  c.Fresh(c.dc, /*salt=*/10, kStressChunk, nUnits);
  ProxyStressKernel<<<kStressBlocks, kStressWarps * 64, 0, c.stream>>>(
      c.dc, c.sendWin, c.recvWin, static_cast<uint32_t*>(c.sendBuf),
      static_cast<ccoGdaSignal_t>(c.nSlots - 1), c.dErr);
  c.Sync();
  return c.Verify("mpsc_stress", 10, kStressChunk, nUnits);
}

static int RunSignalCase(Ctx& c) {
  c.Fresh(c.dc, /*salt=*/0, kChunk, kSpscQueues);
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
  c.bufSize = std::max({nranks * kMaxUnits * kChunk, nranks * kLargeChunk,
                        nranks * kStressBlocks * kStressWarps * kStressRounds * kStressChunk});
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
  ccoDevCommRequirements smallReqs = reqs;
  smallReqs.proxyQueueCapacity = kSmallCapacity;
  if (ccoDevCommCreate(c.comm, &reqs, &c.dc) != 0 ||
      ccoDevCommCreate(c.comm, &smallReqs, &c.dcSmall) != 0) {
    fprintf(stderr, "[rank %d] DevCommCreate failed\n", rank);
    return 1;
  }
  for (const ccoDevComm* d : {&c.dc, &c.dcSmall}) {
    if (d->gdaConnType == CCO_GDA_CONNECTION_NONE || d->proxy.numSpscQueues != kSpscQueues ||
        d->proxy.numMpscQueues != kMpscQueues) {
      fprintf(stderr, "[rank %d] proxy queues missing (gdaConnType=%d spsc=%d mpsc=%d)\n", rank,
              (int)d->gdaConnType, d->proxy.numSpscQueues, d->proxy.numMpscQueues);
      return 1;
    }
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
  fails += RunPutCase<SPSCD2HQueue, ccoCoopWarp, kPutFlushCount>(c, "flush_count", 5, kChunk,
                                                                 kSpscQueues, 64);
  fails += RunPutCase<SPSCD2HQueue, ccoCoopBlock, kPutFlushClobber>(c, "flush_reuse", 6, kChunk,
                                                                    kSpscQueues, 256);
  fails += RunPutCase<SPSCD2HQueue, ccoCoopBlock, kPutKernelPayload>(c, "kernel_payload", 11,
                                                                     kChunk, kSpscQueues, 256);
  fails += RunPutCase<SPSCD2HQueue, ccoCoopWarp, kPutRepeat>(c, "sq_full", 12, kChunk, 1, 64);
  // kMaxUnits blocks over kMpscQueues MPSC queues: several concurrent producers per queue.
  fails += RunPutCase<MPSCD2HQueue, ccoCoopWarp, kPutSignal>(c, "mpsc_put_sig", 7, kChunk,
                                                             kMaxUnits, 64);
  fails += RunPutCase<MPSCD2HQueue, ccoCoopBlock, kPutThenSignal>(c, "mpsc_then_sig", 8, kChunk,
                                                                  kMaxUnits, 256);
  fails += RunPutCase<MPSCD2HQueue, ccoCoopBlock, kPutFlushClobber>(c, "mpsc_flush", 9, kChunk,
                                                                    kMaxUnits, 256);
  fails += RunPutCase<MPSCD2HQueue, ccoCoopBlock, kPutKernelPayload>(c, "mpsc_payload", 13, kChunk,
                                                                     kMaxUnits, 256);
  fails += RunStressCase(c);
  fails += RunSignalCase(c);
  fails += RunPutCase<SPSCD2HQueue, ccoCoopWarp, kPutPieces>(c, "small_spsc", 14, kChunk,
                                                             kSpscQueues, 64, true);
  fails += RunPutCase<MPSCD2HQueue, ccoCoopWarp, kPutPieces>(c, "small_mpsc", 15, kChunk, kMaxUnits,
                                                             64, true);

  HIP_CHECK(hipStreamDestroy(c.stream));
  HIP_CHECK(hipFree(c.dErr));
  ccoDevCommDestroy(c.comm, &c.dcSmall);
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
