// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Johannes Kalmbach <kalmbach@cs.uni-freiburg.de>, UFR

// UFR = University of Freiburg, Chair of Algorithms and Data Structures

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "parser/AsyncFileBlockDriver.h"

#include <boost/asio/use_future.hpp>
#include <memory>
#include <utility>

namespace qlever::parser {

// ____________________________________________________________________________
AsyncFileBlockDriver::AsyncFileBlockDriver(
    const qlever::InputFileSpecification& spec,
    ad_utility::MemorySize blocksize,
    AsyncStatementBoundaryBlockSource::EndPositionFinder findEndPosition,
    std::string description) {
  fileBuffer_ = std::make_unique<AsyncStatementBoundaryBlockSource>(
      ioContext_.get_executor(),
      spec.makeAsyncBlockSource(ioContext_.get_executor(), blocksize),
      std::move(findEndPosition), std::move(description));
  pendingBlock_ = fileBuffer_->asyncGetNextBlock(boost::asio::use_future);
  // Start the I/O thread only after all fallible setup is done (the request
  // queued above simply waits in the `io_context` until the thread picks it
  // up). If the constructor throws before this point, no thread has been
  // started yet, so the implicit member destruction cannot deadlock on the
  // still-engaged work guard; the queued work is destroyed with `ioContext_`.
  ioThread_ = ad_utility::JThread{[this] { ioContext_.run(); }};
}

// ____________________________________________________________________________
AsyncFileBlockDriver::~AsyncFileBlockDriver() {
  // Wait for any in-flight handler to prevent use-after-free of `fileBuffer_`.
  if (pendingBlock_.valid()) {
    pendingBlock_.wait();
  }
  // Controlled teardown: release the work guard so that `ioContext_.run()`
  // returns once the remaining handlers have finished, then join the I/O
  // thread. After the join no thread can touch `fileBuffer_` or the
  // `io_context` anymore, so the members can be destroyed safely (in reverse
  // declaration order, in particular `fileBuffer_` before `ioContext_`).
  workGuard_.reset();
  if (ioThread_.joinable()) {
    ioThread_.join();
  }
}

// ____________________________________________________________________________
std::optional<ByteBlock> AsyncFileBlockDriver::getNextBlock() {
  if (!pendingBlock_.valid()) return std::nullopt;
  auto opt = pendingBlock_.get();
  if (opt.has_value()) {
    pendingBlock_ = fileBuffer_->asyncGetNextBlock(boost::asio::use_future);
  }
  return opt;
}

}  // namespace qlever::parser
