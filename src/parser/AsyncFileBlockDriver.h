// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Johannes Kalmbach <kalmbach@cs.uni-freiburg.de>, UFR

// UFR = University of Freiburg, Chair of Algorithms and Data Structures

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_PARSER_ASYNCFILEBLOCKDRIVER_H
#define QLEVER_SRC_PARSER_ASYNCFILEBLOCKDRIVER_H

#include <boost/asio/executor_work_guard.hpp>
#include <boost/asio/io_context.hpp>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "index/InputFileSpecification.h"
#include "parser/AsyncBlockSource.h"
#include "util/MemorySize/MemorySize.h"
#include "util/jthread.h"

namespace qlever::parser {

// A small wrapper around an `AsyncBlockSource` that is required temporarily
// while the rest of the index builder pipeline is not yet migrated to
// Boost::Asio. It internally holds a
// `unique_ptr<AsyncStatementBoundaryBlockSource>` and schedules it on an
// `io_context` that is driven by a single dedicated I/O thread. The public
// interface is a synchronous `getNextBlock()` function, the asynchronous
// prefetching of the next block is purely internal.
class AsyncFileBlockDriver {
 public:
  // Open the file described by `spec`, wrap it in an
  // `AsyncStatementBoundaryBlockSource` that cuts blocks at the positions
  // determined by `findEndPosition`
  // (`description` is used in error messages, see
  // `AsyncStatementBoundaryBlockSource`), and immediately start prefetching the
  // first block.
  AsyncFileBlockDriver(
      const qlever::InputFileSpecification& spec,
      ad_utility::MemorySize blocksize,
      AsyncStatementBoundaryBlockSource::EndPositionFinder findEndPosition,
      std::string description);

  // Synchronously obtain the next block. This waits for the next block to
  // become available and then immediately starts prefetching the next one.
  std::optional<ByteBlock> getNextBlock();

  ~AsyncFileBlockDriver();

 private:
  // We deliberately do NOT use a `boost::asio::thread_pool` here: its
  // destructor welds `join()` and `shutdown()` together, and on MinGW
  // (winpthreads) `thread_pool::join()` can return before its worker threads
  // have actually left `scheduler::run()`, so the pool's scheduler is freed
  // while a worker is still using it -> use-after-free at teardown (see the
  // identical reasoning in `IndexRebuilder.cpp`). Instead we drive an
  // `io_context` with an explicitly managed worker thread and a teardown whose
  // ordering we fully control (see `~AsyncFileBlockDriver`).
  //
  // The `io_context`, the work guard (which keeps `run()` from returning
  // while the work queue is momentarily empty), and the I/O thread are
  // declared before `fileBuffer_` so that they outlive the source that holds
  // executors of `ioContext_`.
  boost::asio::io_context ioContext_;
  boost::asio::executor_work_guard<boost::asio::io_context::executor_type>
      workGuard_{ioContext_.get_executor()};
  // Started at the end of the constructor (after all fallible setup), joined
  // in the destructor after the work guard has been released.
  ad_utility::JThread ioThread_;
  std::unique_ptr<AsyncBlockSource> fileBuffer_;
  std::future<std::optional<ByteBlock>> pendingBlock_;
};

}  // namespace qlever::parser

#endif  // QLEVER_SRC_PARSER_ASYNCFILEBLOCKDRIVER_H
