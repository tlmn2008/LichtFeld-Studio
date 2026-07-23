/* SPDX-License-Identifier: GPL-3.0-or-later
 * CoreX migration smoke-test support: minimal Logger implementation.
 *
 * The LibTorch-free CUDA tensor kernels reference only lfs::core::Logger::get()
 * and Logger::log(); the real logger.cpp pulls in spdlog (a vcpkg dependency not
 * built on this headless CoreX node). For the on-GPU smoke test we provide a
 * no-op logger: should_emit() (inline in the header) defaults to LogLevel::Info,
 * so the kernels' LOG_DEBUG calls are filtered out before log() is ever reached.
 */
#include "core/logger.hpp"

namespace lfs::core {

    Logger& Logger::get() {
        // Zero-initialised storage: avoids needing the real Logger::Logger()
        // (defined in logger.cpp, which pulls spdlog) and the incomplete Impl
        // destructor. All members are atomics / a null unique_ptr, for which the
        // all-zero bit pattern is a valid state (impl_==nullptr, flags=false).
        alignas(Logger) static unsigned char storage[sizeof(Logger)] = {};
        return *reinterpret_cast<Logger*>(storage);
    }

    void Logger::log(LogLevel /*level*/, const SourceSite& /*loc*/, std::string_view /*msg*/) {
        // no-op: smoke test does not need log output
    }

} // namespace lfs::core
