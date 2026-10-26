/**
 * @file        perf/frame_stats.h
 * @brief       Coarse frame statistics for the [sh-perf] log
 *
 * Copyright (c) 2026, ReXGlue contributors
 * Released under the BSD license - see LICENSE in the root for more details.
 */

#pragma once

#include <cstdint>

namespace rex::perf {

// All of these are cheap ring-buffer style accumulators, safe to call from any
// thread. They do nothing unless the sh_perf_log cvar is enabled.

// One guest frame was submitted (a VdSwap was processed).
void FrameStatsGuestSwap();
// Command processor thread: time spent recording commands for one buffer.
void FrameStatsCommandProcessor(double busy_ms);
// Command processor thread: time blocked waiting for the GPU.
void FrameStatsGpuWait(double wait_ms);
// Command processor thread: total time spent handling one guest frame.
void FrameStatsSwapCost(double swap_ms);
// Host presenter: one vkQueuePresentKHR completed, and how long it took.
void FrameStatsPresent(double present_ms);
// Draw calls issued to the host GPU.
void FrameStatsDraws(uint32_t count);
// Render passes begun and barrier batches submitted by the command processor.
void FrameStatsRenderPass();
void FrameStatsBarrierBatch();
// Time spent executing one PM4 Type 3 opcode, keyed by the opcode itself.
void FrameStatsPacketTime(uint32_t opcode, double ms);
// Whether per-packet timing should be collected at all (two clock queries per
// packet, so it is a separate opt-in on top of the log).
bool FrameStatsPacketTimingEnabled();
// Window paint passes started from the UI thread, and how many of them were
// forced by a window expose event.
void FrameStatsPaint(bool forced_by_expose);

enum class PaintRequestSource : uint32_t {
  // A guest frame was refreshed and needs to be shown.
  kGuestOutput = 0,
  // A UI drawer (ImGui) asked to be redrawn.
  kUIDrawer = 1,
  // The surface, its monitor or its size changed.
  kSurfaceChange = 2,
  // A paint requested itself another paint at the end of the pass.
  kSelfTick = 3,
  kSelfImmediate = 4,
  // Any other path, counted where the window paint event is actually queued.
  kOther = 5,
  // Requests deferred to the end of the current paint pass.
  kUIDrawerInPaint = 6,
  kDrawersChangeInPaint = 7,
};
void FrameStatsPaintRequest(PaintRequestSource source);
// Host submissions (vkQueueSubmit of a command buffer).
void FrameStatsSubmit(double cost_ms);
// Time the command processor waited for the host queue (held by the presenter).
void FrameStatsQueueWait(double wait_ms);

// RAII timer reporting the elapsed time to one of the accumulators.
class FrameStatsScope {
 public:
  enum class Sink {
    kSwapCost,
    kCommandProcessor,
    kGpuWait,
    // Sub-divisions of the command processor time.
    kDraw,
    kBarriers,
    // Pipeline cache work done on the command processor thread.
    kShaderTranslate,
    kPlaceholderPipeline,
    kSyncPipeline,
  };

  explicit FrameStatsScope(Sink sink);
  ~FrameStatsScope();

  FrameStatsScope(const FrameStatsScope&) = delete;
  FrameStatsScope& operator=(const FrameStatsScope&) = delete;

 private:
  Sink sink_;
  uint64_t start_tick_;
  bool active_;
};

// Per-frame event counters for the [sh-perf] "gpu memory" line.
enum class FrameStatsCounter : uint32_t {
  kUploads,
  kUploadBytes,
  kInvalidations,
  kInvalidatedBytes,
  kIndexConversions,
  kIndexConvertedBytes,
  // Upload overhead breakdown for the [sh-perf] "upload overhead" line. The
  // *Ticks counters hold host clock ticks.
  kUploadCalls,
  kProtectCalls,
  kProtectTicks,
  kGuestFaults,
  kGuestFaultTicks,
  kLockWaitTicks,
  kCount,
};
void FrameStatsCount(FrameStatsCounter counter, uint64_t value);

// RAII timer adding the elapsed host ticks to a *Ticks counter.
class FrameStatsCounterTimer {
 public:
  explicit FrameStatsCounterTimer(FrameStatsCounter counter);
  ~FrameStatsCounterTimer();

  FrameStatsCounterTimer(const FrameStatsCounterTimer&) = delete;
  FrameStatsCounterTimer& operator=(const FrameStatsCounterTimer&) = delete;

 private:
  FrameStatsCounter counter_;
  uint64_t start_tick_;
  bool active_;
};

// Splits the time of one draw into phases for the [sh-perf] "draw phases"
// line. Mark(p) charges the time since the previous mark to phase p; the
// destructor charges the rest to the last phase.
class FrameStatsDrawPhases {
 public:
  enum Phase : uint32_t {
    kPrimitive,
    kShaders,
    kTextures,
    kRenderTargets,
    kPipeline,
    kBindings,
    kVertexMemory,
    kSubmit,
    kCount,
  };

  FrameStatsDrawPhases();
  ~FrameStatsDrawPhases();

  void Mark(Phase phase);

  FrameStatsDrawPhases(const FrameStatsDrawPhases&) = delete;
  FrameStatsDrawPhases& operator=(const FrameStatsDrawPhases&) = delete;

 private:
  uint64_t last_tick_;
  bool active_;
};

}  // namespace rex::perf
