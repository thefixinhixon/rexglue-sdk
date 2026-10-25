/**
 * @file        core/perf/frame_stats.cpp
 * @brief       Coarse frame statistics for the [sh-perf] log
 *
 * Emits one line every sh_perf_log_interval_ms:
 *
 *   [sh-perf] 2.00s guest 61.2/s host 61.2/s presents 122 drift 0
 *             swap 2.41ms present 0.31ms cp 1.83ms/frame gpu_wait 0.0ms/frame
 *             draws 1348/frame
 *
 * "guest" is the VdSwap rate (frames the game submitted), "host" the
 * vkQueuePresentKHR rate. "swap" is the per-guest-frame cost on the command
 * processor thread, "cp" the command recording part of it, "gpu_wait" the time
 * that thread spent blocked on GPU fences. With software presentation (headless
 * MESA_VK_WSI_DEBUG=sw) "present" is not representative of a real GPU.
 *
 * Copyright (c) 2026, ReXGlue contributors
 * Released under the BSD license - see LICENSE in the root for more details.
 */
#include <rex/perf/frame_stats.h>

#include <atomic>
#include <cinttypes>

#include <rex/chrono/clock.h>
#include <rex/cvar.h>
#include <rex/logging.h>

REXCVAR_DEFINE_BOOL(sh_perf_log, false, "Performance",
                    "Log guest/host frame rates, command processor time and GPU wait time");
REXCVAR_DEFINE_INT32(sh_perf_log_interval_ms, 2000, "Performance",
                     "Interval between [sh-perf] statistics lines, in milliseconds")
    .range(100, 60000);

namespace rex::perf {

namespace {

struct Accumulators {
  std::atomic<uint64_t> guest_swaps{0};
  std::atomic<uint64_t> presents{0};
  std::atomic<uint64_t> draws{0};
  std::atomic<uint64_t> cp_us{0};
  std::atomic<uint64_t> gpu_wait_us{0};
  std::atomic<uint64_t> swap_us{0};
  std::atomic<uint64_t> present_us{0};
};

Accumulators g_acc;

std::atomic<bool> g_enabled{false};
std::atomic<uint64_t> g_last_log_tick{0};
std::atomic<uint64_t> g_total_presents{0};

uint64_t ToMicroseconds(uint64_t ticks) {
  uint64_t freq = rex::chrono::Clock::QueryHostTickFrequency();
  if (!freq) {
    return 0;
  }
  return ticks * 1000000ull / freq;
}

void MaybeLog() {
  static std::atomic<bool> logging{false};
  if (!g_enabled.load(std::memory_order_relaxed)) {
    return;
  }
  if (logging.exchange(true, std::memory_order_acquire)) {
    return;
  }
  uint64_t now = rex::chrono::Clock::QueryHostTickCount();
  uint64_t last = g_last_log_tick.load(std::memory_order_relaxed);
  uint64_t interval_us = static_cast<uint64_t>(REXCVAR_GET(sh_perf_log_interval_ms)) * 1000ull;
  if (!last) {
    g_last_log_tick.store(now, std::memory_order_relaxed);
    logging.store(false, std::memory_order_release);
    return;
  }
  uint64_t elapsed_us = ToMicroseconds(now - last);
  if (elapsed_us < interval_us) {
    logging.store(false, std::memory_order_release);
    return;
  }
  g_last_log_tick.store(now, std::memory_order_relaxed);

  uint64_t guest_swaps = g_acc.guest_swaps.exchange(0, std::memory_order_relaxed);
  uint64_t presents = g_acc.presents.exchange(0, std::memory_order_relaxed);
  uint64_t draws = g_acc.draws.exchange(0, std::memory_order_relaxed);
  uint64_t cp_us = g_acc.cp_us.exchange(0, std::memory_order_relaxed);
  uint64_t gpu_wait_us = g_acc.gpu_wait_us.exchange(0, std::memory_order_relaxed);
  uint64_t swap_us = g_acc.swap_us.exchange(0, std::memory_order_relaxed);
  uint64_t present_us = g_acc.present_us.exchange(0, std::memory_order_relaxed);
  logging.store(false, std::memory_order_release);

  double seconds = static_cast<double>(elapsed_us) / 1000000.0;
  double frames = static_cast<double>(guest_swaps ? guest_swaps : 1);
  REXLOG_INFO(
      "[sh-perf] {:.2f}s guest {:.1f}/s host {:.1f}/s | swap {:.2f}ms present {:.2f}ms "
      "cp {:.2f}ms/frame gpu_wait {:.2f}ms/frame | draws {:.0f}/frame",
      seconds, static_cast<double>(guest_swaps) / seconds,
      static_cast<double>(presents) / seconds,
      static_cast<double>(swap_us) / frames / 1000.0,
      presents ? static_cast<double>(present_us) / static_cast<double>(presents) / 1000.0 : 0.0,
      static_cast<double>(cp_us) / frames / 1000.0,
      static_cast<double>(gpu_wait_us) / frames / 1000.0,
      static_cast<double>(draws) / frames);
}

void RefreshEnabled() {
  bool enabled = REXCVAR_GET(sh_perf_log);
  if (!enabled && g_enabled.load(std::memory_order_relaxed)) {
    g_last_log_tick.store(0, std::memory_order_relaxed);
  }
  g_enabled.store(enabled, std::memory_order_relaxed);
}

struct EnabledCheck {
  bool value;
  EnabledCheck() {
    RefreshEnabled();
    value = g_enabled.load(std::memory_order_relaxed);
  }
};

}  // namespace

void FrameStatsGuestSwap() {
  EnabledCheck enabled;
  if (!enabled.value) {
    return;
  }
  g_acc.guest_swaps.fetch_add(1, std::memory_order_relaxed);
  MaybeLog();
}

void FrameStatsCommandProcessor(double busy_ms) {
  if (!g_enabled.load(std::memory_order_relaxed)) {
    return;
  }
  g_acc.cp_us.fetch_add(static_cast<uint64_t>(busy_ms * 1000.0), std::memory_order_relaxed);
}

void FrameStatsGpuWait(double wait_ms) {
  if (!g_enabled.load(std::memory_order_relaxed)) {
    return;
  }
  g_acc.gpu_wait_us.fetch_add(static_cast<uint64_t>(wait_ms * 1000.0), std::memory_order_relaxed);
}

void FrameStatsSwapCost(double swap_ms) {
  if (!g_enabled.load(std::memory_order_relaxed)) {
    return;
  }
  g_acc.swap_us.fetch_add(static_cast<uint64_t>(swap_ms * 1000.0), std::memory_order_relaxed);
}

void FrameStatsPresent(double present_ms) {
  EnabledCheck enabled;
  if (!enabled.value) {
    return;
  }
  g_acc.presents.fetch_add(1, std::memory_order_relaxed);
  if (present_ms > 0.0) {
    g_acc.present_us.fetch_add(static_cast<uint64_t>(present_ms * 1000.0),
                               std::memory_order_relaxed);
  }
  MaybeLog();
}

void FrameStatsDraws(uint32_t count) {
  if (!g_enabled.load(std::memory_order_relaxed)) {
    return;
  }
  g_acc.draws.fetch_add(count, std::memory_order_relaxed);
}

FrameStatsScope::FrameStatsScope(Sink sink) : sink_(sink) {
  RefreshEnabled();
  active_ = g_enabled.load(std::memory_order_relaxed);
  start_tick_ = active_ ? rex::chrono::Clock::QueryHostTickCount() : 0;
}

FrameStatsScope::~FrameStatsScope() {
  if (!active_) {
    return;
  }
  double milliseconds =
      static_cast<double>(ToMicroseconds(rex::chrono::Clock::QueryHostTickCount() - start_tick_)) /
      1000.0;
  switch (sink_) {
    case Sink::kSwapCost:
      FrameStatsSwapCost(milliseconds);
      break;
    case Sink::kCommandProcessor:
      FrameStatsCommandProcessor(milliseconds);
      break;
    case Sink::kGpuWait:
      FrameStatsGpuWait(milliseconds);
      break;
  }
}

}  // namespace rex::perf
