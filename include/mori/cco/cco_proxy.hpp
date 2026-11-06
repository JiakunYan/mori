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
//
// cco_proxy.hpp — ccoProxy: host-proxy RDMA transport with the ccoGda window API.
//
// Kernels push fixed-size descriptors into a KIWI device-to-host queue (the
// GPU->CPU queue layer); host proxy threads pop them and post ibverbs work
// requests on host-created RC QPs, one per (proxy thread, peer), on the PD that
// owns the cco window MRs. Keys and remote addresses are resolved on the GPU
// exactly like ccoGda (iova=0: raddr = window offset), so the proxy needs no key
// tables.
//
// Queues: ccoProxy<SPSCD2HQueue> (default) indexes the DevComm's SPSC pool,
// ccoProxy<MPSCD2HQueue> its MPSC pool. Only one wavefront may use an SPSC queue
// at a time: its producer state lives in device memory, loaded by the
// constructor and written back by the destructor (or release()).
//
// Ordering: ops pushed by one wavefront to one queue are popped in program
// order and land on one QP per peer, so a put followed by a signal (or a put
// carrying a signal) to the same peer through the same queue is delivered in
// order -- the same guarantee ccoGda gives within one context. Different queues
// are unordered, as different ccoGda contexts are.
//
// Remote signals are NIC atomic fetch-adds into the same slots ccoGda uses, so
// readSignal/waitSignal/resetSignal read devComm.ibgda.signalBuf and need a
// DevComm whose gdaConnectionType resolves to non-NONE (FULL / RAIL /
// CROSSNODE).
//
// flush<Mode>() pushes a FLUSH descriptor at queue position P and spins until
// the proxy has published flushDone >= P:
//   ccoProxyFlushKiwi (default): everything before the FLUSH in this queue has
//     been popped and posted. Source buffers are NOT yet reusable.
//   ccoProxyFlushNetwork: additionally every work request this queue posted
//     before the FLUSH has completed (remote ack): source buffers are reusable.
// flush(peer) is flush() (covers every peer).
//
// Requires a BUILD_CCO_PROXY build and a wave64 GFX9 target (KIWI push).
//
// Current limits, both pending KIWI queue APIs:
//   - Coop must be ccoCoopWarp or ccoCoopBlock: a KIWI push is
//     wavefront-collective (all 64 lanes). ccoCoopThread needs a per-thread
//     push (push_thread()).
//   - flush() is SPSC only. On an MPSC queue the producer cannot tell where
//     its FLUSH landed in the shared ring; that needs a push that reports the
//     end of its claimed lines (push_lanes_end()).

#pragma once

#include "mori/cco/cco_scale_out.hpp"

#if defined(__HIPCC__)
#include "kiwi/queue/device_to_host_queue.hip.hpp"
#include "kiwi/queue/mpsc_device_to_host_queue.hip.hpp"
#endif

namespace mori {
namespace cco {

/* ────────────────────────────────────────────────────────────────────────────
 *  Wire format (shared by the device API and the host proxy engine)
 *
 *  KIWI metadata (32 bits) = op | worldPeer << 8. Payload is one of the
 *  descriptors below; the op enum stays open for later ops (e.g. write-flag
 *  signals).
 * ──────────────────────────────────────────────────────────────────────────── */

enum ccoProxyOp : uint32_t {
  ccoProxyOpPut = 1,        // ccoProxyPutDesc: RDMA write
  ccoProxyOpPutSignal = 2,  // ccoProxyPutSignalDesc: RDMA write + atomic add, same QP
  ccoProxyOpSignal = 3,     // ccoProxySignalDesc: atomic add
  ccoProxyOpFlush = 4,      // ccoProxyFlushDesc: publish flushDone (peer unused)
};

enum ccoProxyFlushMode : uint32_t {
  ccoProxyFlushKiwi = 0,     // satisfied when the proxy pops the FLUSH
  ccoProxyFlushNetwork = 1,  // ... and the queue's earlier work requests completed
};

struct ccoProxyPutDesc {
  uint64_t laddr;  // local window offset (iova=0)
  uint64_t raddr;  // remote window offset (iova=0)
  uint32_t lkey;
  uint32_t rkey;
  uint32_t bytes;
  uint32_t pad;
};

struct ccoProxySignalDesc {
  uint64_t raddr;  // remote signal slot (window offset, iova=0)
  uint64_t value;  // atomic add operand
  uint32_t rkey;
  uint32_t pad;
};

struct ccoProxyPutSignalDesc {
  ccoProxyPutDesc put;
  ccoProxySignalDesc signal;
};

struct ccoProxyFlushDesc {
  uint32_t mode;  // ccoProxyFlushMode
  uint32_t pad;
};

static_assert(sizeof(ccoProxyPutDesc) == 32, "one queue line");
static_assert(sizeof(ccoProxySignalDesc) == 24, "one queue line");
static_assert(sizeof(ccoProxyPutSignalDesc) == 56, "two queue lines");
static_assert(sizeof(ccoProxyFlushDesc) == 8, "one queue line");

// Largest single RDMA write a descriptor carries; put() splits larger transfers.
inline constexpr uint64_t ccoProxyMaxPutBytes = 1ull << 30;

CCO_HOST_DEVICE_INLINE uint32_t ccoProxyMakeMeta(uint32_t op, int worldPeer) {
  return op | (static_cast<uint32_t>(worldPeer) << 8);
}
CCO_HOST_DEVICE_INLINE uint32_t ccoProxyMetaOp(uint32_t meta) { return meta & 0xffu; }
CCO_HOST_DEVICE_INLINE int ccoProxyMetaPeer(uint32_t meta) { return static_cast<int>(meta >> 8); }

#if defined(__HIPCC__)

namespace impl {

// Broadcast lane 0's copy of a trivially copyable value to the whole wavefront.
// The KIWI push needs identical arguments on every lane; like ccoGda, the
// lane-0 (thread_rank 0) arguments are the ones that count.
template <typename T>
__device__ inline T proxyBroadcast(const T& v) {
  static_assert(std::is_trivially_copyable_v<T> && sizeof(T) % sizeof(uint32_t) == 0,
                "proxyBroadcast needs a trivially copyable, 4-byte-multiple type");
  struct Words {
    uint32_t w[sizeof(T) / sizeof(uint32_t)];
  };
  Words words = __builtin_bit_cast(Words, v);
  for (size_t i = 0; i < sizeof(T) / sizeof(uint32_t); i++) {
    words.w[i] = __builtin_amdgcn_readfirstlane(words.w[i]);
  }
  return __builtin_bit_cast(T, words);
}

// Resolve a remote action to a signal descriptor, as ccoGda::put does.
template <typename RemoteAction>
__device__ inline ccoProxySignalDesc proxyResolveSignal(ccoDevComm const& comm, int worldPeer,
                                                        const RemoteAction& action) {
  ccoProxySignalDesc s{};
  if constexpr (std::is_same_v<RemoteAction, ccoGda_SignalInc>) {
    s.raddr = static_cast<uint64_t>(action.signalId) * sizeof(uint64_t);
    s.rkey = comm.resourceWindow_inlined.ibgdaWin.peerRkeys[worldPeer];
    s.value = 1;
  } else if constexpr (std::is_same_v<RemoteAction, ccoGda_SignalAdd>) {
    s.raddr = static_cast<uint64_t>(action.signalId) * sizeof(uint64_t);
    s.rkey = comm.resourceWindow_inlined.ibgdaWin.peerRkeys[worldPeer];
    s.value = action.value;
  } else if constexpr (std::is_same_v<RemoteAction, ccoGda_WindowSignalAdd>) {
    // Caller's window, offset already window-relative (iova=0).
    s.raddr = action.offset;
    s.rkey = reinterpret_cast<ccoWindowDevice*>(action.win)->ibgdaWin.peerRkeys[worldPeer];
    s.value = action.value;
  } else {
    static_assert(std::is_same_v<RemoteAction, ccoGda_NoSignal>,
                  "ccoProxy supports ccoGda_NoSignal / SignalInc / SignalAdd / WindowSignalAdd");
  }
  return s;
}

}  // namespace impl

// Device-side transport handle. Construct per thread; call put / signal /
// flush from the whole Coop group. Not copyable: an SPSC handle writes its
// producer state back exactly once.
template <typename Queue = kiwi::queue::SPSCD2HQueue>
struct ccoProxy {
  static constexpr bool kSpsc = std::is_same_v<Queue, kiwi::queue::SPSCD2HQueue>;
  static_assert(kSpsc || std::is_same_v<Queue, kiwi::queue::MPSCD2HQueue>,
                "ccoProxy queue must be kiwi::queue::SPSCD2HQueue or MPSCD2HQueue");

  ccoDevComm const& comm;
  int rank;    // my index in the GDA team [0, nRanks)
  int nRanks;  // GDA team size, as ccoGda
  uint32_t queueId;
  uint64_t* flushDone;      // this queue's flushDone (fine-grained device memory)
  uint64_t* producerState;  // SPSC: persisted {producer_index, consumer_index_cached}
  bool dirty;               // SPSC: pushed since the state was loaded / written back
  typename Queue::DeviceHandle queue;

  // queueIndex is taken modulo the pool size, like ccoGda's contextIndex.
  __device__ inline ccoProxy(ccoDevComm const& comm_, int queueIndex);
  __device__ inline ~ccoProxy() { release(); }
  ccoProxy(const ccoProxy&) = delete;
  ccoProxy& operator=(const ccoProxy&) = delete;

  // SPSC: write the producer state back so the next ccoProxy on this queue
  // (later in the kernel or in a later launch) continues from it. Called by
  // the destructor; a no-op for MPSC and for wavefronts that never pushed.
  __device__ inline void release() {
    if constexpr (kSpsc) {
      if (dirty && __lane_id() == 0) {
        __hip_atomic_store(&producerState[0], queue.producer_index, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_store(&producerState[1], queue.consumer_index_cached, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
      }
      dirty = false;
    }
  }

  template <ccoTeamMode TeamMode>
  __device__ inline int resolveWorldPeer(int peer) const {
    if constexpr (TeamMode == CCO_TEAM_WORLD) {
      return peer;
    } else {
      return impl::GdaPeerToWorld(comm, peer);
    }
  }

  // RDMA write of [srcWin+srcOffset, +bytes) to peer's [dstWin+dstOffset, ...),
  // optionally followed (same QP, so after the data) by a remote atomic signal.
  // Template parameters follow ccoGda::put; ThreadMode only applies to
  // ccoCoopThread, which is not supported yet (see the file comment).
  template <ccoTeamMode TeamMode = CCO_TEAM_WORLD,
            ccoGdaThreadMode ThreadMode = ccoGdaThreadIndependent,
            typename RemoteAction = ccoGda_NoSignal, typename Coop = ccoCoopWarp>
  __device__ inline void put(int peer, ccoWindow_t dstWin, size_t dstOffset, ccoWindow_t srcWin,
                             size_t srcOffset, size_t bytes, RemoteAction remoteAction = {},
                             Coop coop = Coop{});

  // Remote atomic signal only.
  template <ccoTeamMode TeamMode = CCO_TEAM_WORLD, typename RemoteAction = ccoGda_SignalInc,
            typename Coop = ccoCoopWarp>
  __device__ inline void signal(int peer, RemoteAction remoteAction, Coop coop = Coop{});

  // Push FLUSH{Mode} and wait until the proxy has satisfied it (see the file
  // comment). Waits only for this queue.
  template <ccoProxyFlushMode Mode = ccoProxyFlushKiwi, typename Coop = ccoCoopWarp,
            std::enable_if_t<!std::is_integral_v<Coop>, int> = 0>
  __device__ inline void flush(Coop coop = Coop{});

  // v1: identical to flush() -- covers every peer.
  template <ccoProxyFlushMode Mode = ccoProxyFlushKiwi, ccoTeamMode TeamMode = CCO_TEAM_WORLD,
            typename Coop = ccoCoopWarp>
  __device__ inline void flush(int peer, Coop coop = Coop{}) {
    (void)peer;
    flush<Mode>(coop);
  }

  // Local signal slots: identical to ccoGda (remote NIC atomics land there).
  __device__ inline uint64_t readSignal(ccoGdaSignal_t signalId, int bits = 64) {
    return impl::readSignalImpl<CCO_GDA_BUILD_PROVIDER>(comm.ibgda.signalBuf,
                                                        comm.ibgda.signalShadows, signalId, bits);
  }
  template <typename Coop = ccoCoopThread>
  __device__ inline void waitSignal(ccoGdaSignal_t signalId, uint64_t least, Coop coop = Coop{},
                                    int bits = 64) {
    coop.sync();
    if (coop.thread_rank() == 0) {
      impl::waitSignalImpl<CCO_GDA_BUILD_PROVIDER>(comm.ibgda.signalBuf, comm.ibgda.signalShadows,
                                                   signalId, least, bits);
    }
    coop.sync();
  }
  __device__ inline void resetSignal(ccoGdaSignal_t signalId) {
    impl::resetSignalImpl<CCO_GDA_BUILD_PROVIDER>(comm.ibgda.signalBuf, comm.ibgda.signalShadows,
                                                  signalId);
  }

 private:
  template <typename Coop>
  __device__ static inline bool issuingWave() {
    static_assert(!std::is_same_v<Coop, ccoCoopThread>,
                  "ccoProxy put/signal/flush need ccoCoopWarp or ccoCoopBlock: a KIWI queue push "
                  "is wavefront-collective. ccoCoopThread needs a per-thread KIWI push "
                  "(push_thread()), not available yet.");
    static_assert(std::is_same_v<Coop, ccoCoopWarp> || std::is_same_v<Coop, ccoCoopBlock>,
                  "ccoProxy supports ccoCoopWarp and ccoCoopBlock");
    if constexpr (std::is_same_v<Coop, ccoCoopBlock>) {
      // Wavefront 0. (Not impl::warpSize(): core/utils.hpp, pulled in by
      // cco_scale_out.hpp, defines a `warpSize` macro.)
      return impl::threadIdxX() < static_cast<unsigned>(__builtin_amdgcn_wavefrontsize());
    } else {
      return true;  // the calling wavefront
    }
  }

  // Push one descriptor (whole wavefront). An SPSC push on a full ring
  // retries; an MPSC push waits inside KIWI (and always succeeds).
  template <typename Desc>
  __device__ inline void push(uint32_t op, int worldPeer, const Desc& desc) {
    const uint32_t meta = ccoProxyMakeMeta(op, worldPeer);
    if constexpr (kSpsc) {
      while (!queue.push_reg(desc, meta)) {
        __builtin_amdgcn_s_sleep(1);
      }
      dirty = true;
    } else {
      queue.push_reg(desc, meta);
    }
  }

  // Data-carrying push: make the source data (and any earlier stores the peer
  // will read) visible to the NIC before the proxy can post the WR.
  template <typename Desc>
  __device__ inline void pushOp(uint32_t op, int worldPeer, const Desc& desc) {
    impl::threadFenceSystem();
    push(op, worldPeer, desc);
  }
};

template <typename Queue>
__device__ inline ccoProxy<Queue>::ccoProxy(ccoDevComm const& comm_, int queueIndex)
    : comm(comm_), flushDone(nullptr), producerState(nullptr), dirty(false) {
  switch (comm.gdaConnType) {
    case CCO_GDA_CONNECTION_FULL:
      rank = comm.rank;
      nRanks = comm.worldSize;
      break;
    case CCO_GDA_CONNECTION_RAIL:
      rank = comm.rank / comm.lsaSize;
      nRanks = comm.worldSize / comm.lsaSize;
      break;
    case CCO_GDA_CONNECTION_CROSSNODE:
      rank = (comm.rank / comm.lsaSize) * comm.lsaSize;
      nRanks = comm.worldSize - comm.lsaSize + 1;
      break;
    default:
      rank = 0;
      nRanks = 0;
      break;
  }
  const int poolSize = kSpsc ? comm.proxy.numSpscQueues : comm.proxy.numMpscQueues;
  const ccoProxyQueueRaw* pool = kSpsc ? comm.proxy.spscQueues : comm.proxy.mpscQueues;
  queueId = poolSize > 0 ? static_cast<uint32_t>(queueIndex) % poolSize : 0;
  queue = {};
  if (poolSize > 0) {
    namespace kq = kiwi::queue::detail;
    const ccoProxyQueueRaw& raw = pool[queueId];
    flushDone = raw.flushDone;
    producerState = raw.producerState;
    queue.chunks = (kq::global_ptr<kq::Chunk>)raw.chunks;
    queue.consumer_index = (kq::global_ptr<const uint64_t>)raw.consumerIndex;
    queue.log2_capacity = raw.log2Capacity;
    if constexpr (kSpsc) {
      queue.producer_index =
          __hip_atomic_load(&producerState[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      queue.consumer_index_cached =
          __hip_atomic_load(&producerState[1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } else {
      queue.producer_index = (kq::global_ptr<uint64_t>)raw.producerState;
      queue.consumer_index_cached = 0;  // any lower bound is safe; refreshed on demand
    }
  }
}

template <typename Queue>
template <ccoTeamMode TeamMode, ccoGdaThreadMode ThreadMode, typename RemoteAction, typename Coop>
__device__ inline void ccoProxy<Queue>::put(int peer, ccoWindow_t dstWin, size_t dstOffset,
                                            ccoWindow_t srcWin, size_t srcOffset, size_t bytes,
                                            RemoteAction remoteAction, Coop coop) {
  coop.sync();
  if (issuingWave<Coop>()) {
    peer = impl::proxyBroadcast(peer);
    dstWin = impl::proxyBroadcast(dstWin);
    srcWin = impl::proxyBroadcast(srcWin);
    dstOffset = impl::proxyBroadcast(dstOffset);
    srcOffset = impl::proxyBroadcast(srcOffset);
    bytes = impl::proxyBroadcast(bytes);
    if constexpr (!std::is_same_v<RemoteAction, ccoGda_NoSignal>) {
      remoteAction = impl::proxyBroadcast(remoteAction);
    }
    const int worldPeer = resolveWorldPeer<TeamMode>(peer);

    ccoProxyPutDesc d{};
    d.lkey = reinterpret_cast<ccoWindowDevice*>(srcWin)->ibgdaWin.lkey;
    d.rkey = reinterpret_cast<ccoWindowDevice*>(dstWin)->ibgdaWin.peerRkeys[worldPeer];
    d.laddr = srcOffset;
    d.raddr = dstOffset;
    // Split transfers one WR cannot carry; the signal rides the last piece.
    while (bytes > ccoProxyMaxPutBytes) {
      d.bytes = static_cast<uint32_t>(ccoProxyMaxPutBytes);
      pushOp(ccoProxyOpPut, worldPeer, d);
      d.laddr += ccoProxyMaxPutBytes;
      d.raddr += ccoProxyMaxPutBytes;
      bytes -= ccoProxyMaxPutBytes;
    }
    d.bytes = static_cast<uint32_t>(bytes);
    if constexpr (std::is_same_v<RemoteAction, ccoGda_NoSignal>) {
      pushOp(ccoProxyOpPut, worldPeer, d);
    } else {
      ccoProxyPutSignalDesc ds{};
      ds.put = d;
      ds.signal = impl::proxyResolveSignal(comm, worldPeer, remoteAction);
      pushOp(ccoProxyOpPutSignal, worldPeer, ds);
    }
  }
  coop.sync();
}

template <typename Queue>
template <ccoTeamMode TeamMode, typename RemoteAction, typename Coop>
__device__ inline void ccoProxy<Queue>::signal(int peer, RemoteAction remoteAction, Coop coop) {
  static_assert(!std::is_same_v<RemoteAction, ccoGda_NoSignal>, "signal() needs a remote action");
  coop.sync();
  if (issuingWave<Coop>()) {
    peer = impl::proxyBroadcast(peer);
    remoteAction = impl::proxyBroadcast(remoteAction);
    const int worldPeer = resolveWorldPeer<TeamMode>(peer);
    pushOp(ccoProxyOpSignal, worldPeer, impl::proxyResolveSignal(comm, worldPeer, remoteAction));
  }
  coop.sync();
}

template <typename Queue>
template <ccoProxyFlushMode Mode, typename Coop, std::enable_if_t<!std::is_integral_v<Coop>, int>>
__device__ inline void ccoProxy<Queue>::flush(Coop coop) {
  static_assert(kSpsc,
                "ccoProxy<MPSCD2HQueue>::flush is not available yet: it needs the position of its "
                "FLUSH in the shared ring, which needs a KIWI MPSC push that reports it "
                "(push_lanes_end()). Use the SPSC pool to flush.");
  coop.sync();
  if (issuingWave<Coop>()) {
    ccoProxyFlushDesc d{};
    d.mode = Mode;
    push(ccoProxyOpFlush, 0, d);
    // Position just past our FLUSH; flushDone reaches it only once the proxy
    // has satisfied this FLUSH (it retires flushes in queue order).
    const uint64_t pos = queue.producer_index;
    if (__lane_id() == 0) {
      while (__hip_atomic_load(flushDone, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < pos) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
    // Later stores (e.g. overwriting a source buffer) stay after the wait.
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
  }
  coop.sync();
}

#endif  // defined(__HIPCC__)

}  // namespace cco
}  // namespace mori
