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

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cinttypes>

#include <rex/chrono/clock.h>
#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/platform.h>

REXCVAR_DEFINE_BOOL(sh_perf_log, false, "Performance",
                    "Log guest/host frame rates, command processor time and GPU wait time");
REXCVAR_DEFINE_INT32(sh_perf_log_interval_ms, 2000, "Performance",
                     "Interval between [sh-perf] statistics lines, in milliseconds")
    .range(100, 60000);
REXCVAR_DEFINE_BOOL(sh_perf_log_packets, false, "Performance",
                    "Also time every PM4 packet for [sh-perf] (adds measurable overhead)");

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
  std::atomic<uint64_t> paints{0};
  std::atomic<uint64_t> expose_paints{0};
  std::atomic<uint64_t> submits{0};
  std::atomic<uint64_t> submit_us{0};
  std::atomic<uint64_t> queue_wait_us{0};
  std::atomic<uint64_t> paint_requests[8];
  std::atomic<uint64_t> render_passes{0};
  std::atomic<uint64_t> barrier_batches{0};
  std::atomic<uint64_t> draw_us{0};
  std::atomic<uint64_t> barrier_us{0};
  // Pipeline cache work on the command processor thread: translations,
  // placeholder pipelines, pipelines created synchronously.
  std::atomic<uint64_t> pipeline_count[3];
  std::atomic<uint64_t> pipeline_us[3];
  std::atomic<uint64_t> pipeline_max_us[3];
  std::atomic<uint64_t> draw_phase_ticks[FrameStatsDrawPhases::kCount];
  std::atomic<uint64_t> counters[uint32_t(FrameStatsCounter::kCount)];
  // Type 3 opcode is 7 bits wide.
  std::atomic<uint64_t> packet_us[128];
};

Accumulators g_acc;

std::atomic<bool> g_enabled{false};
std::atomic<uint64_t> g_last_log_tick{0};

size_t CountMappings() {
#if REX_PLATFORM_GNU_LINUX
  FILE* maps = std::fopen("/proc/self/maps", "r");
  if (!maps) {
    return 0;
  }
  size_t lines = 0;
  char buffer[65536];
  size_t read;
  while ((read = std::fread(buffer, 1, sizeof(buffer), maps)) != 0) {
    lines += size_t(std::count(buffer, buffer + read, '\n'));
  }
  std::fclose(maps);
  return lines;
#else
  return 0;
#endif
}

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
  uint64_t paints = g_acc.paints.exchange(0, std::memory_order_relaxed);
  uint64_t expose_paints = g_acc.expose_paints.exchange(0, std::memory_order_relaxed);
  uint64_t submits = g_acc.submits.exchange(0, std::memory_order_relaxed);
  uint64_t submit_us = g_acc.submit_us.exchange(0, std::memory_order_relaxed);
  uint64_t queue_wait_us = g_acc.queue_wait_us.exchange(0, std::memory_order_relaxed);
  uint64_t req_guest =
      g_acc.paint_requests[0].exchange(0, std::memory_order_relaxed);
  uint64_t req_ui = g_acc.paint_requests[1].exchange(0, std::memory_order_relaxed);
  uint64_t req_surface = g_acc.paint_requests[2].exchange(0, std::memory_order_relaxed);
  uint64_t req_self_tick = g_acc.paint_requests[3].exchange(0, std::memory_order_relaxed);
  uint64_t req_self_imm = g_acc.paint_requests[4].exchange(0, std::memory_order_relaxed);
  uint64_t req_other = g_acc.paint_requests[5].exchange(0, std::memory_order_relaxed);
  uint64_t req_ui_paint = g_acc.paint_requests[6].exchange(0, std::memory_order_relaxed);
  uint64_t req_drawers_paint = g_acc.paint_requests[7].exchange(0, std::memory_order_relaxed);
  uint64_t render_passes = g_acc.render_passes.exchange(0, std::memory_order_relaxed);
  uint64_t barrier_batches = g_acc.barrier_batches.exchange(0, std::memory_order_relaxed);
  uint64_t draw_us = g_acc.draw_us.exchange(0, std::memory_order_relaxed);
  uint64_t barrier_us = g_acc.barrier_us.exchange(0, std::memory_order_relaxed);
  uint64_t packet_us[128];
  for (uint32_t i = 0; i < 128; ++i) {
    packet_us[i] = g_acc.packet_us[i].exchange(0, std::memory_order_relaxed);
  }
  uint64_t pipeline_count[3], pipeline_us[3], pipeline_max_us[3];
  for (uint32_t i = 0; i < 3; ++i) {
    pipeline_count[i] = g_acc.pipeline_count[i].exchange(0, std::memory_order_relaxed);
    pipeline_us[i] = g_acc.pipeline_us[i].exchange(0, std::memory_order_relaxed);
    pipeline_max_us[i] = g_acc.pipeline_max_us[i].exchange(0, std::memory_order_relaxed);
  }
  logging.store(false, std::memory_order_release);

  double seconds = static_cast<double>(elapsed_us) / 1000000.0;
  double frames = static_cast<double>(guest_swaps ? guest_swaps : 1);
  REXLOG_INFO(
      "[sh-perf] {:.2f}s guest {:.1f}/s host {:.1f}/s paints {:.1f}/s expose {:.1f}/s | "
      "req g{:.1f} u{:.1f} s{:.1f} st{:.1f} si{:.1f} o{:.1f} up{:.1f} dp{:.1f} | "
      "swap {:.2f}ms present {:.2f}ms cp {:.2f}ms/frame gpu_wait {:.2f}ms/frame | "
      "draws {:.0f}/frame passes {:.0f} barriers {:.0f} submits {:.1f}/frame "
      "submit {:.2f}ms qwait {:.2f}ms/frame | draw {:.2f}ms bar {:.2f}ms",
      seconds, static_cast<double>(guest_swaps) / seconds,
      static_cast<double>(presents) / seconds, static_cast<double>(paints) / seconds,
      static_cast<double>(expose_paints) / seconds,
      static_cast<double>(req_guest) / seconds, static_cast<double>(req_ui) / seconds,
      static_cast<double>(req_surface) / seconds, static_cast<double>(req_self_tick) / seconds,
      static_cast<double>(req_self_imm) / seconds, static_cast<double>(req_other) / seconds,
      static_cast<double>(req_ui_paint) / seconds,
      static_cast<double>(req_drawers_paint) / seconds,
      static_cast<double>(swap_us) / frames / 1000.0,
      presents ? static_cast<double>(present_us) / static_cast<double>(presents) / 1000.0 : 0.0,
      static_cast<double>(cp_us) / frames / 1000.0,
      static_cast<double>(gpu_wait_us) / frames / 1000.0,
      static_cast<double>(draws) / frames, static_cast<double>(render_passes) / frames,
      static_cast<double>(barrier_batches) / frames, static_cast<double>(submits) / frames,
      submits ? static_cast<double>(submit_us) / static_cast<double>(submits) / 1000.0 : 0.0,
      static_cast<double>(queue_wait_us) / frames / 1000.0,
      static_cast<double>(draw_us) / frames / 1000.0,
      static_cast<double>(barrier_us) / frames / 1000.0);

  // The most expensive packet types, as a second line to keep the first one
  // readable. Opcode numbers are the PM4 Type 3 opcodes (hex).
  std::string packets;
  for (uint32_t entry = 0; entry < 5; ++entry) {
    uint32_t best = 128;
    for (uint32_t i = 0; i < 128; ++i) {
      if (!packet_us[i]) {
        continue;
      }
      if (best == 128 || packet_us[i] > packet_us[best]) {
        best = i;
      }
    }
    if (best == 128) {
      break;
    }
    packets += fmt::format("{}{:#04x} {:.2f}ms", entry ? " " : "", best,
                           static_cast<double>(packet_us[best]) / frames / 1000.0);
    packet_us[best] = 0;
  }
  if (!packets.empty()) {
    REXLOG_INFO("[sh-perf] cp packets per frame: {}", packets);
  }
  uint64_t phase_ticks[FrameStatsDrawPhases::kCount];
  uint64_t phase_total = 0;
  for (uint32_t i = 0; i < FrameStatsDrawPhases::kCount; ++i) {
    phase_ticks[i] = g_acc.draw_phase_ticks[i].exchange(0, std::memory_order_relaxed);
    phase_total += phase_ticks[i];
  }
  if (phase_total) {
    auto phase_ms = [&](uint32_t i) {
      return static_cast<double>(ToMicroseconds(phase_ticks[i])) / frames / 1000.0;
    };
    REXLOG_INFO(
        "[sh-perf] draw phases per frame: primitive {:.2f}ms shaders {:.2f}ms textures {:.2f}ms "
        "rt {:.2f}ms pipeline {:.2f}ms bindings {:.2f}ms vmem {:.2f}ms submit {:.2f}ms",
        phase_ms(0), phase_ms(1), phase_ms(2), phase_ms(3), phase_ms(4), phase_ms(5), phase_ms(6),
        phase_ms(7));
  }
  uint64_t counters[uint32_t(FrameStatsCounter::kCount)];
  for (uint32_t i = 0; i < uint32_t(FrameStatsCounter::kCount); ++i) {
    counters[i] = g_acc.counters[i].exchange(0, std::memory_order_relaxed);
  }
  REXLOG_INFO(
      "[sh-perf] gpu memory per frame: uploads {:.1f} {:.1f}KB | invalidations {:.1f} {:.1f}KB | "
      "index conversions {:.1f} {:.1f}KB",
      counters[0] / frames, counters[1] / frames / 1024.0, counters[2] / frames,
      counters[3] / frames / 1024.0, counters[4] / frames, counters[5] / frames / 1024.0);
  auto counter_ms = [&](FrameStatsCounter c) {
    return static_cast<double>(ToMicroseconds(counters[uint32_t(c)])) / frames / 1000.0;
  };
  auto counter_per_frame = [&](FrameStatsCounter c) { return counters[uint32_t(c)] / frames; };
  REXLOG_INFO(
      "[sh-perf] upload overhead per frame: upload calls {:.1f} | protect {:.1f} calls {:.2f}ms | "
      "guest faults {:.1f} {:.2f}ms | lock wait {:.2f}ms | vmas {}",
      counter_per_frame(FrameStatsCounter::kUploadCalls),
      counter_per_frame(FrameStatsCounter::kProtectCalls),
      counter_ms(FrameStatsCounter::kProtectTicks),
      counter_per_frame(FrameStatsCounter::kGuestFaults),
      counter_ms(FrameStatsCounter::kGuestFaultTicks),
      counter_ms(FrameStatsCounter::kLockWaitTicks), CountMappings());
  if (pipeline_count[0] || pipeline_count[1] || pipeline_count[2]) {
    REXLOG_INFO(
        "[sh-perf] cp pipeline work: translate {} {:.1f}ms (max {:.1f}) | placeholder {} "
        "{:.1f}ms (max {:.1f}) | sync create {} {:.1f}ms (max {:.1f})",
        pipeline_count[0], pipeline_us[0] / 1000.0, pipeline_max_us[0] / 1000.0,
        pipeline_count[1], pipeline_us[1] / 1000.0, pipeline_max_us[1] / 1000.0,
        pipeline_count[2], pipeline_us[2] / 1000.0, pipeline_max_us[2] / 1000.0);
  }
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

void FrameStatsPaint(bool forced_by_expose) {
  if (!g_enabled.load(std::memory_order_relaxed)) {
    return;
  }
  g_acc.paints.fetch_add(1, std::memory_order_relaxed);
  if (forced_by_expose) {
    g_acc.expose_paints.fetch_add(1, std::memory_order_relaxed);
  }
}

void FrameStatsPaintRequest(PaintRequestSource source) {
  if (!g_enabled.load(std::memory_order_relaxed)) {
    return;
  }
  g_acc.paint_requests[static_cast<uint32_t>(source)].fetch_add(1, std::memory_order_relaxed);
}

void FrameStatsRenderPass() {
  if (!g_enabled.load(std::memory_order_relaxed)) {
    return;
  }
  g_acc.render_passes.fetch_add(1, std::memory_order_relaxed);
}

void FrameStatsBarrierBatch() {
  if (!g_enabled.load(std::memory_order_relaxed)) {
    return;
  }
  g_acc.barrier_batches.fetch_add(1, std::memory_order_relaxed);
}

bool FrameStatsPacketTimingEnabled() {
  return g_enabled.load(std::memory_order_relaxed) && REXCVAR_GET(sh_perf_log_packets);
}

void FrameStatsPacketTime(uint32_t opcode, double ms) {
  if (!g_enabled.load(std::memory_order_relaxed)) {
    return;
  }
  g_acc.packet_us[opcode & 127].fetch_add(static_cast<uint64_t>(ms * 1000.0),
                                          std::memory_order_relaxed);
}

void FrameStatsQueueWait(double wait_ms) {
  if (!g_enabled.load(std::memory_order_relaxed)) {
    return;
  }
  g_acc.queue_wait_us.fetch_add(static_cast<uint64_t>(wait_ms * 1000.0), std::memory_order_relaxed);
}

void FrameStatsSubmit(double cost_ms) {
  if (!g_enabled.load(std::memory_order_relaxed)) {
    return;
  }
  g_acc.submits.fetch_add(1, std::memory_order_relaxed);
  g_acc.submit_us.fetch_add(static_cast<uint64_t>(cost_ms * 1000.0), std::memory_order_relaxed);
}

void FrameStatsCount(FrameStatsCounter counter, uint64_t value) {
  if (!g_enabled.load(std::memory_order_relaxed)) {
    return;
  }
  g_acc.counters[uint32_t(counter)].fetch_add(value, std::memory_order_relaxed);
}

FrameStatsCounterTimer::FrameStatsCounterTimer(FrameStatsCounter counter) : counter_(counter) {
  active_ = g_enabled.load(std::memory_order_relaxed);
  start_tick_ = active_ ? rex::chrono::Clock::QueryHostTickCount() : 0;
}

FrameStatsCounterTimer::~FrameStatsCounterTimer() {
  if (!active_) {
    return;
  }
  g_acc.counters[uint32_t(counter_)].fetch_add(
      rex::chrono::Clock::QueryHostTickCount() - start_tick_, std::memory_order_relaxed);
}

FrameStatsDrawPhases::FrameStatsDrawPhases() {
  active_ = g_enabled.load(std::memory_order_relaxed);
  last_tick_ = active_ ? rex::chrono::Clock::QueryHostTickCount() : 0;
}

FrameStatsDrawPhases::~FrameStatsDrawPhases() {
  Mark(kSubmit);
}

void FrameStatsDrawPhases::Mark(Phase phase) {
  if (!active_) {
    return;
  }
  uint64_t now = rex::chrono::Clock::QueryHostTickCount();
  g_acc.draw_phase_ticks[phase].fetch_add(now - last_tick_, std::memory_order_relaxed);
  last_tick_ = now;
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
    case Sink::kDraw:
      g_acc.draw_us.fetch_add(static_cast<uint64_t>(milliseconds * 1000.0),
                              std::memory_order_relaxed);
      break;
    case Sink::kShaderTranslate:
    case Sink::kPlaceholderPipeline:
    case Sink::kSyncPipeline: {
      uint32_t i = uint32_t(sink_) - uint32_t(Sink::kShaderTranslate);
      uint64_t us = static_cast<uint64_t>(milliseconds * 1000.0);
      g_acc.pipeline_count[i].fetch_add(1, std::memory_order_relaxed);
      g_acc.pipeline_us[i].fetch_add(us, std::memory_order_relaxed);
      uint64_t prev = g_acc.pipeline_max_us[i].load(std::memory_order_relaxed);
      while (us > prev &&
             !g_acc.pipeline_max_us[i].compare_exchange_weak(prev, us, std::memory_order_relaxed)) {
      }
      break;
    }
    case Sink::kBarriers:
      g_acc.barrier_us.fetch_add(static_cast<uint64_t>(milliseconds * 1000.0),
                                 std::memory_order_relaxed);
      break;
  }
}

}  // namespace rex::perf
