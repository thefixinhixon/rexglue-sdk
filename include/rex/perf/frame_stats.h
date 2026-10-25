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

// RAII timer reporting the elapsed time to one of the accumulators.
class FrameStatsScope {
 public:
  enum class Sink {
    kSwapCost,
    kCommandProcessor,
    kGpuWait,
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

}  // namespace rex::perf
