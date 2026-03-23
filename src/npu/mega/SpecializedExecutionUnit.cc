/*
 * Copyright (c) 2026
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "npu/mega/SpecializedExecutionUnit.hh"

#include <algorithm>
#include <cstring>

#include "base/trace.hh"
#include "debug/SpecializedExecutionUnit.hh"
#include "sim/system.hh"

namespace gem5
{

namespace
{

constexpr Addr SyncIndicatorBase = 0x71000000;
constexpr Addr DefaultSpmBase = 0x60000000;
constexpr Addr DefaultSpmSlotStride = 0x40;
constexpr size_t MultiportReadMaskWord = 1;
constexpr size_t MultiportWriteMaskWord = 2;
constexpr size_t MultiportRepetitionWord = 3;
constexpr size_t MultiportReservedWord = 4;

SpecializedExecutionUnit::MemTxnContext::Kind
inferTxnKind(PacketPtr pkt)
{
    if (pkt->isRead()) {
        return SpecializedExecutionUnit::MemTxnContext::Kind::Mvin;
    }

    if (pkt->isWrite() && pkt->getAddr() == SyncIndicatorBase) {
        return SpecializedExecutionUnit::MemTxnContext::Kind::SyncWrite;
    }

    return SpecializedExecutionUnit::MemTxnContext::Kind::Mvout;
}

PacketPtr
buildPacketFromRequest(const RequestPtr &req,
                       const SpecializedExecutionUnit::MemRequestDesc &desc)
{
    const MemCmd cmd =
        desc.kind == SpecializedExecutionUnit::MemTxnContext::Kind::Mvin ?
        MemCmd::ReadReq : MemCmd::WriteReq;
    PacketPtr pkt = new Packet(req, cmd);
    pkt->allocate();

    if (desc.kind == SpecializedExecutionUnit::MemTxnContext::Kind::Mvout &&
        !desc.data.empty()) {
        pkt->setData(desc.data.data());
    }

    return pkt;
}

uint32_t
readCmdWord(const std::vector<uint8_t> &cmd, size_t word_idx)
{
    const size_t offset = word_idx * sizeof(uint32_t);
    if (cmd.size() < offset + sizeof(uint32_t)) {
        return 0;
    }

    uint32_t word = 0;
    std::memcpy(&word, cmd.data() + offset, sizeof(word));
    return word;
}

std::vector<uint8_t>
packWord(uint32_t value)
{
    std::vector<uint8_t> bytes(sizeof(value), 0);
    std::memcpy(bytes.data(), &value, sizeof(value));
    return bytes;
}

uint32_t
unpackWord(const std::vector<uint8_t> &bytes)
{
    uint32_t value = 0;
    if (!bytes.empty()) {
        std::memcpy(&value, bytes.data(),
                    std::min(bytes.size(), sizeof(value)));
    }
    return value;
}

uint32_t
defaultInitialSlotValue(PortID port_id)
{
    return 0x00010011U + (static_cast<uint32_t>(port_id) * 0x00011111U);
}

bool
maskFitsPortCount(uint32_t mask, size_t num_ports)
{
    if (num_ports >= 32) {
        return true;
    }

    const uint32_t valid_mask = (1U << num_ports) - 1U;
    return (mask & ~valid_mask) == 0;
}

} // anonymous namespace

SpecializedExecutionUnit::CPUSidePort::CPUSidePort(
    const std::string &name, SpecializedExecutionUnit *owner)
    : ResponsePort(name, owner),
      owner(owner),
      needRetry(false),
      blockedRespPacket(nullptr),
      sendResponseEvent([this] { sendDeferredResponse(); }, name)
{
}

void
SpecializedExecutionUnit::CPUSidePort::trySendRetry()
{
    if (needRetry && blockedRespPacket == nullptr) {
        needRetry = false;
        sendRetryReq();
    }
}

void
SpecializedExecutionUnit::CPUSidePort::sendDeferredResponse()
{
    if (blockedRespPacket == nullptr) {
        return;
    }

    PacketPtr pkt = blockedRespPacket;
    blockedRespPacket = nullptr;

    if (!sendTimingResp(pkt)) {
        blockedRespPacket = pkt;
        return;
    }

    trySendRetry();
}

bool
SpecializedExecutionUnit::CPUSidePort::recvTimingReq(PacketPtr pkt)
{
    if (blockedRespPacket != nullptr || needRetry) {
        needRetry = true;
        return false;
    }

    if (!owner->handleRequest(pkt)) {
        needRetry = true;
        return false;
    }

    if (pkt->needsResponse()) {
        pkt->makeResponse();
        blockedRespPacket = pkt;
        if (!sendResponseEvent.scheduled()) {
            owner->schedule(sendResponseEvent, owner->clockEdge(Cycles(1)));
        }
    }

    return true;
}

void
SpecializedExecutionUnit::CPUSidePort::recvRespRetry()
{
    sendDeferredResponse();
}

AddrRangeList
SpecializedExecutionUnit::CPUSidePort::getAddrRanges() const
{
    return owner->getAddrRanges();
}

SpecializedExecutionUnit::MemSidePort::MemSidePort(
    const std::string &name, SpecializedExecutionUnit *owner)
    : RequestPort(name, owner), owner(owner), blockedPacket(nullptr)
{
}

void
SpecializedExecutionUnit::MemSidePort::sendPacket(PacketPtr pkt)
{
    panic_if(blockedPacket != nullptr,
             "Should never try to send if blocked!");

    if (!sendTimingReq(pkt)) {
        blockedPacket = pkt;
    }
}

bool
SpecializedExecutionUnit::MemSidePort::recvTimingResp(PacketPtr pkt)
{
    return owner->handleMemResponse(pkt);
}

void
SpecializedExecutionUnit::MemSidePort::recvReqRetry()
{
    assert(blockedPacket != nullptr);
    PacketPtr pkt = blockedPacket;
    blockedPacket = nullptr;
    sendPacket(pkt);
}

SpecializedExecutionUnit::SpecializedExecutionUnit(
    const SpecializedExecutionUnitParams &params)
    : ClockedObject(params),
      cpuSidePort(params.name + ".cpu_side", this),
      memSidePort(params.name + ".mem_side_legacy", this),
      macroCmdBytes(params.macro_cmd_bytes),
      cmdQueueDepth(params.cmd_queue_depth),
      baseAddr(params.base_addr),
      syncEnqueueOnDataWrite(params.sync_enqueue_on_data_write),
      debugProcessLatency(params.debug_process_latency),
      issueCmdBusy(false),
      completedCount(0),
      activeMemPacket(nullptr),
      activeCmd(macroCmdBytes, 0),
      issueEvent([this] { issueOneCommand(); }, name() + ".issueEvent"),
      finishExecutionEvent([this] { finishExecution(); },
                           name() + ".finishExecutionEvent")
{
    panic_if(params.num_mem_side_ports == 0,
             "SpecializedExecutionUnit requires at least one mem_side port");

    stagingBuffer.bytes.resize(macroCmdBytes, 0);
    memSidePorts.reserve(params.num_mem_side_ports);
    for (PortID i = 0; i < params.num_mem_side_ports; ++i) {
        auto port = std::make_unique<MemSidePort>(
            csprintf("%s.mem_side[%d]", params.name, i), this);
        port->portId = i;
        memSidePorts.push_back(std::move(port));
    }

    DPRINTF(SpecializedExecutionUnit,
            "Created SEU: base_addr=%#x cmd_bytes=%u queue_depth=%u "
            "mem_ports=%zu\n",
            baseAddr, macroCmdBytes, cmdQueueDepth, memSidePorts.size());
}

SpecializedExecutionUnit::~SpecializedExecutionUnit()
{
    cleanupActiveMemPacket();
}

void
SpecializedExecutionUnit::init()
{
    ClockedObject::init();

    if (cpuSidePort.isConnected()) {
        cpuSidePort.sendRangeChange();
    }
}

void
SpecializedExecutionUnit::cleanupActiveMemPacket()
{
    if (activeMemPacket) {
        delete activeMemPacket;
        activeMemPacket = nullptr;
    }

    for (auto &[pkt, txn] : activeMemTxns) {
        if (pkt != activeMemPacket) {
            delete pkt;
        }
    }
    activeMemTxns.clear();
}

Port &
SpecializedExecutionUnit::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "cpu_side") {
        return cpuSidePort;
    } else if (if_name == "mem_side") {
        if (idx == InvalidPortID) {
            idx = 0;
        }
        return getMemSidePort(idx);
    }
    return ClockedObject::getPort(if_name, idx);
}

SpecializedExecutionUnit::MemSidePort &
SpecializedExecutionUnit::getMemSidePort(PortID idx)
{
    panic_if(idx < 0 || static_cast<size_t>(idx) >= memSidePorts.size(),
             "%s: mem_side port index %d out of range (num ports=%zu)",
             name(), idx, memSidePorts.size());
    return *memSidePorts[idx];
}

const SpecializedExecutionUnit::MemSidePort &
SpecializedExecutionUnit::getMemSidePort(PortID idx) const
{
    panic_if(idx < 0 || static_cast<size_t>(idx) >= memSidePorts.size(),
             "%s: mem_side port index %d out of range (num ports=%zu)",
             name(), idx, memSidePorts.size());
    return *memSidePorts[idx];
}

AddrRangeList
SpecializedExecutionUnit::getAddrRanges() const
{
    AddrRangeList ranges;
    ranges.push_back(AddrRange(baseAddr, baseAddr + 2 * macroCmdBytes));
    return ranges;
}

bool
SpecializedExecutionUnit::validMmioOffset(Addr offset, size_t size) const
{
    return offset + size <= 2 * macroCmdBytes;
}

bool
SpecializedExecutionUnit::writeDataBytes(Addr offset, const uint8_t *src,
                                          size_t size)
{
    if (!validMmioOffset(offset, size)) {
        return false;
    }
    for (size_t i = 0; i < size; ++i) {
        stagingBuffer.bytes[offset + i] = src[i];
    }
    return true;
}

bool
SpecializedExecutionUnit::writeDataChunk(Addr offset, PacketPtr pkt)
{
    size_t size = pkt->getSize();
    if (!validMmioOffset(offset, size)) {
        return false;
    }
    const uint8_t *data = pkt->getConstPtr<uint8_t>();
    return writeDataBytes(offset, data, size);
}

bool
SpecializedExecutionUnit::canLaunchCmd() const
{
    return cmdQueue.size() < cmdQueueDepth;
}

bool
SpecializedExecutionUnit::launchStagedCmd()
{
    if (!canLaunchCmd()) {
        DPRINTF(SpecializedExecutionUnit,
                "Launch rejected: queue full (%zu/%u)\n",
                cmdQueue.size(), cmdQueueDepth);
        return false;
    }
    cmdQueue.push_back(stagingBuffer.bytes);
    DPRINTF(SpecializedExecutionUnit,
            "Launched command, queue size now %zu\n", cmdQueue.size());
    tryScheduleIssue();
    return true;
}

bool
SpecializedExecutionUnit::handleRequest(PacketPtr pkt)
{
    if (!pkt->isWrite()) {
        panic("SpecializedExecutionUnit only accepts write requests");
    }

    Addr addr = pkt->getAddr();
    Addr offset = addr - baseAddr;

    if (!validMmioOffset(offset, pkt->getSize())) {
        panic("MMIO write out of bounds: addr=%#x offset=%#x size=%u",
              addr, offset, pkt->getSize());
    }

    DPRINTF(SpecializedExecutionUnit,
            "MMIO write: addr=%#x offset=%#x size=%u\n",
            addr, offset, pkt->getSize());

    bool success = true;
    if (offset < macroCmdBytes) {
        // Staging area write
        writeDataChunk(offset, pkt);

        if (syncEnqueueOnDataWrite && offset == 0) {
            success = launchStagedCmd();
        }
    } else {
        // Control area write - trigger launch
        success = launchStagedCmd();
    }

    // Response will be sent in the next cycle by CPUSidePort::recvTimingReq
    // Do not send response here

    return success;
}

void
SpecializedExecutionUnit::tryScheduleIssue()
{
    if (!issueEvent.scheduled() && !cmdQueue.empty() && !issueCmdBusy) {
        schedule(issueEvent, nextCycle());
    }
}

void
SpecializedExecutionUnit::issueOneCommand()
{
    if (issueCmdBusy || cmdQueue.empty()) {
        return;
    }

    std::vector<uint8_t> cmd = cmdQueue.front();
    cmdQueue.pop_front();
    activeCmd = cmd;
    issueCmdBusy = true;

    DPRINTF(SpecializedExecutionUnit,
            "Issuing command, queue size now %zu\n", cmdQueue.size());

    startExecuteCommand(activeCmd);
}

void
SpecializedExecutionUnit::startExecuteCommand(const std::vector<uint8_t> &cmd)
{
    activeExecution.cmd = cmd;
    beginActiveCommand(activeExecution);
    advanceActivePhase(activeExecution);
}

void
SpecializedExecutionUnit::completeActiveCommand()
{
    issueCmdBusy = false;

    uint32_t syncWord = 0;
    const bool sentCompletionSync =
        buildCompletionSyncWord(activeCmd, syncWord);
    if (sentCompletionSync) {
        sendCompletionSyncWord(syncWord);
    }

    completedCount++;

    DPRINTF(SpecializedExecutionUnit,
            "Finished command %lu, queue size %zu\n",
            completedCount, cmdQueue.size());

    if (!sentCompletionSync && !cmdQueue.empty()) {
        schedule(issueEvent, nextCycle());
    }

    // If we previously blocked a launch due to full queue, retry now
    cpuSidePort.trySendRetry();
}

void
SpecializedExecutionUnit::finishExecution()
{
    if (activeExecution.phase == Phase::Executing) {
        finishExecutePhase(activeExecution);
        return;
    }

    completeActiveCommand();
}

Tick
SpecializedExecutionUnit::process(const std::vector<uint8_t> &cmd)
{
    DPRINTF(SpecializedExecutionUnit,
            "Processing command, latency=%lu ticks\n", debugProcessLatency);
    return debugProcessLatency;
}

void
SpecializedExecutionUnit::beginActiveCommand(ActiveExecution &exec)
{
    const uint64_t total_prologues = exec.prologueCount;
    const uint64_t total_executes = exec.executeCount;
    const uint64_t total_epilogues = exec.epilogueCount;
    const uint64_t total_reads = exec.completedReadRespCount;
    const uint64_t total_writes = exec.completedWriteRespCount;
    const uint64_t total_iterations = exec.completedIterations;

    exec = ActiveExecution{};
    exec.cmd = activeCmd;
    exec.fields = parseCmdFields(extractCmdWord(exec.cmd));
    exec.phase = Phase::Prologue;
    exec.prologueCount = total_prologues;
    exec.executeCount = total_executes;
    exec.epilogueCount = total_epilogues;
    exec.completedReadRespCount = total_reads;
    exec.completedWriteRespCount = total_writes;
    exec.completedIterations = total_iterations;

    exec.readMask = readCmdWord(exec.cmd, MultiportReadMaskWord);
    exec.writeMask = readCmdWord(exec.cmd, MultiportWriteMaskWord);
    exec.repetition = readCmdWord(exec.cmd, MultiportRepetitionWord);
    exec.reserved = readCmdWord(exec.cmd, MultiportReservedWord);

    panic_if(!maskFitsPortCount(exec.readMask, memSidePorts.size()),
             "%s: read mask %#x exceeds mem port count %zu",
             name(), exec.readMask, memSidePorts.size());
    panic_if(!maskFitsPortCount(exec.writeMask, memSidePorts.size()),
             "%s: write mask %#x exceeds mem port count %zu",
             name(), exec.writeMask, memSidePorts.size());
    if (exec.repetition == 0) {
        exec.repetition = 1;
    }

    for (PortID port = 0; port < static_cast<PortID>(memSidePorts.size());
         ++port) {
        exec.readResults[port] = packWord(defaultInitialSlotValue(port));
    }

    onCommandBegin(exec);
}

void
SpecializedExecutionUnit::advanceActivePhase(ActiveExecution &exec)
{
    switch (exec.phase) {
      case Phase::Prologue:
        runProloguePhase(exec);
        break;
      case Phase::LaunchingMvin:
        launchMvinPhase(exec);
        break;
      case Phase::LaunchingMvout:
        launchMvoutPhase(exec);
        break;
      case Phase::Epilogue:
        runEpiloguePhase(exec);
        break;
      default:
        panic("%s: invalid phase transition request %d",
              name(), static_cast<int>(exec.phase));
    }
}

void
SpecializedExecutionUnit::runProloguePhase(ActiveExecution &exec)
{
    exec.readPorts.clear();
    exec.writePorts.clear();
    exec.writeResults.clear();
    exec.prologueCount++;

    if (exec.repetition == 0) {
        exec.repetition = 1;
    }

    for (PortID port = 0; port < static_cast<PortID>(memSidePorts.size());
         ++port) {
        const uint32_t bit = 1U << port;
        if ((exec.readMask & bit) != 0) {
            exec.readPorts.push_back(port);
        }
        if ((exec.writeMask & bit) != 0) {
            exec.writePorts.push_back(port);
        }
    }

    prologue(exec);

    exec.phase = Phase::LaunchingMvin;
    advanceActivePhase(exec);
}

void
SpecializedExecutionUnit::launchMvinPhase(ActiveExecution &exec)
{
    std::vector<MemRequestDesc> reqs;
    buildMvinRequests(exec, reqs);

    if (reqs.empty()) {
        for (const PortID port : exec.readPorts) {
            MemRequestDesc req;
            req.portId = port;
            req.kind = MemTxnContext::Kind::Mvin;
            req.addr = DefaultSpmBase +
                       (static_cast<Addr>(port) * DefaultSpmSlotStride);
            req.size = sizeof(uint32_t);
            reqs.push_back(req);
        }
    }

    if (reqs.empty()) {
        scheduleExecutePhase(exec);
        return;
    }

    exec.phase = Phase::WaitingMvin;
    for (auto &req_desc : reqs) {
        req_desc.kind = MemTxnContext::Kind::Mvin;
        req_desc.iteration = exec.iteration;
        req_desc.token = nextMemTxnToken++;

        RequestPtr req = std::make_shared<Request>(
            req_desc.addr, req_desc.size, Request::Flags(),
            Request::funcRequestorId);
        PacketPtr pkt = buildPacketFromRequest(req, req_desc);

        MemTxnContext txn;
        txn.pkt = pkt;
        txn.portId = req_desc.portId;
        txn.kind = req_desc.kind;
        txn.iteration = req_desc.iteration;
        txn.token = req_desc.token;
        txn.addr = req_desc.addr;
        txn.size = req_desc.size;

        auto [it, inserted] = activeMemTxns.emplace(pkt, txn);
        panic_if(!inserted, "%s: duplicate in-flight packet registration",
                 name());
        getMemSidePort(req_desc.portId).sendPacket(pkt);
    }
}

bool
SpecializedExecutionUnit::handleMvinResponseInternal(ActiveExecution &exec,
                                                      PacketPtr pkt)
{
    auto it = activeMemTxns.find(pkt);
    panic_if(it == activeMemTxns.end(),
             "%s: missing mvin transaction for packet addr=%#x",
             name(), pkt->getAddr());

    const MemTxnContext txn = it->second;
    const uint8_t *data = pkt->getConstPtr<uint8_t>();
    exec.readResults[txn.portId] =
        std::vector<uint8_t>(data, data + pkt->getSize());
    exec.completedReadRespCount++;
    onMvinResponse(exec, txn, pkt);

    activeMemTxns.erase(it);
    delete pkt;

    if (exec.phase == Phase::WaitingMvin) {
        size_t remaining = 0;
        for (const auto &[active_pkt, active_txn] : activeMemTxns) {
            if (active_txn.kind == MemTxnContext::Kind::Mvin &&
                active_txn.iteration == exec.iteration) {
                remaining++;
            }
        }
        if (remaining == 0) {
            scheduleExecutePhase(exec);
        }
    }
    return true;
}

void
SpecializedExecutionUnit::scheduleExecutePhase(ActiveExecution &exec)
{
    exec.phase = Phase::Executing;
    exec.executeCount++;
    const Tick execLatency = execute(exec);
    schedule(finishExecutionEvent, curTick() + execLatency);
}

void
SpecializedExecutionUnit::finishExecutePhase(ActiveExecution &exec)
{
    exec.phase = Phase::LaunchingMvout;
    advanceActivePhase(exec);
}

void
SpecializedExecutionUnit::launchMvoutPhase(ActiveExecution &exec)
{
    std::vector<MemRequestDesc> reqs;
    buildMvoutRequests(exec, reqs);

    if (reqs.empty()) {
        uint32_t signature = 0;
        for (const PortID port : exec.readPorts) {
            signature += unpackWord(exec.readResults[port]);
        }
        for (const PortID port : exec.writePorts) {
            if (exec.writeResults.find(port) == exec.writeResults.end()) {
                const uint32_t current = unpackWord(exec.readResults[port]);
                const uint32_t value = current + signature +
                    ((static_cast<uint32_t>(exec.iteration) + 1U) * 0x10U) +
                    (static_cast<uint32_t>(port) + 1U);
                exec.writeResults[port] = packWord(value);
            }
            MemRequestDesc req;
            req.portId = port;
            req.kind = MemTxnContext::Kind::Mvout;
            req.addr = DefaultSpmBase +
                       (static_cast<Addr>(port) * DefaultSpmSlotStride);
            req.size = sizeof(uint32_t);
            req.data = exec.writeResults[port];
            reqs.push_back(req);
        }
    }

    if (reqs.empty()) {
        exec.phase = Phase::Epilogue;
        advanceActivePhase(exec);
        return;
    }

    exec.phase = Phase::WaitingMvout;
    for (auto &req_desc : reqs) {
        req_desc.kind = MemTxnContext::Kind::Mvout;
        req_desc.iteration = exec.iteration;
        req_desc.token = nextMemTxnToken++;

        RequestPtr req = std::make_shared<Request>(
            req_desc.addr, req_desc.size, Request::Flags(),
            Request::funcRequestorId);
        PacketPtr pkt = buildPacketFromRequest(req, req_desc);

        MemTxnContext txn;
        txn.pkt = pkt;
        txn.portId = req_desc.portId;
        txn.kind = req_desc.kind;
        txn.iteration = req_desc.iteration;
        txn.token = req_desc.token;
        txn.addr = req_desc.addr;
        txn.size = req_desc.size;

        auto [it, inserted] = activeMemTxns.emplace(pkt, txn);
        panic_if(!inserted, "%s: duplicate in-flight packet registration",
                 name());
        getMemSidePort(req_desc.portId).sendPacket(pkt);
    }
}

bool
SpecializedExecutionUnit::handleMvoutResponseInternal(ActiveExecution &exec,
                                                       PacketPtr pkt)
{
    auto it = activeMemTxns.find(pkt);
    panic_if(it == activeMemTxns.end(),
             "%s: missing mvout transaction for packet addr=%#x",
             name(), pkt->getAddr());

    const MemTxnContext txn = it->second;
    exec.completedWriteRespCount++;
    if (auto write_it = exec.writeResults.find(txn.portId);
        write_it != exec.writeResults.end()) {
        exec.readResults[txn.portId] = write_it->second;
    }
    onMvoutResponse(exec, txn, pkt);

    activeMemTxns.erase(it);
    delete pkt;

    if (exec.phase == Phase::WaitingMvout) {
        size_t remaining = 0;
        for (const auto &[active_pkt, active_txn] : activeMemTxns) {
            if (active_txn.kind == MemTxnContext::Kind::Mvout &&
                active_txn.iteration == exec.iteration) {
                remaining++;
            }
        }
        if (remaining == 0) {
            exec.phase = Phase::Epilogue;
            advanceActivePhase(exec);
        }
    }
    return true;
}

void
SpecializedExecutionUnit::runEpiloguePhase(ActiveExecution &exec)
{
    exec.epilogueCount++;
    exec.completedIterations++;
    epilogue(exec);

    ActiveExecution exit_view = exec;
    exit_view.completedIterations = exec.iteration + 1;
    if (shouldExit(exit_view)) {
        finalizeActiveCommand(exec);
        return;
    }

    exec.iteration++;
    exec.phase = Phase::Prologue;
    advanceActivePhase(exec);
}

void
SpecializedExecutionUnit::finalizeActiveCommand(ActiveExecution &exec)
{
    exec.phase = Phase::Completing;
    completeActiveCommand();
    exec.phase = Phase::Idle;
}

void
SpecializedExecutionUnit::sendMemRequest(PacketPtr pkt)
{
    sendMemRequest(pkt, 0);
}

void
SpecializedExecutionUnit::sendMemRequest(PacketPtr pkt, PortID port_id)
{
    MemTxnContext txn;
    txn.pkt = pkt;
    txn.portId = port_id;
    txn.kind = inferTxnKind(pkt);
    txn.iteration = activeExecution.iteration;
    txn.token = nextMemTxnToken++;
    txn.addr = pkt->getAddr();
    txn.size = pkt->getSize();

    auto [it, inserted] = activeMemTxns.emplace(pkt, txn);
    panic_if(!inserted, "%s: duplicate in-flight packet registration", name());

    DPRINTF(SpecializedExecutionUnit,
            "sendMemRequest: port=%d kind=%d addr=%#x size=%u inflight=%zu\n",
            port_id, static_cast<int>(txn.kind), txn.addr, txn.size,
            activeMemTxns.size());

    getMemSidePort(port_id).sendPacket(pkt);
}

bool
SpecializedExecutionUnit::buildCompletionSyncWord(
    const std::vector<uint8_t> &cmd, uint32_t &word) const
{
    const CmdFields fields = parseCmdFields(extractCmdWord(cmd));
    if (!fields.setIndicatorSns && !fields.setIndicatorSnd) {
        return false;
    }

    word = (static_cast<uint32_t>(0x1U) << 28) |
           (static_cast<uint32_t>(fields.deviceId) << 24) |
           (static_cast<uint32_t>(1U) << 16) |
           (static_cast<uint32_t>(fields.syncIndicator) << 8);
    return true;
}

void
SpecializedExecutionUnit::sendCompletionSyncWord(uint32_t word)
{
    sendCompletionSyncWord(word, 0);
}

void
SpecializedExecutionUnit::sendCompletionSyncWord(uint32_t word, PortID port_id)
{
    RequestPtr req = std::make_shared<Request>(
        SyncIndicatorBase, sizeof(uint32_t), Request::Flags(),
        Request::funcRequestorId);
    PacketPtr pkt = new Packet(req, MemCmd::WriteReq);
    pkt->allocate();
    pkt->setData(reinterpret_cast<const uint8_t *>(&word));

    DPRINTF(
        SpecializedExecutionUnit,
        "completion sync write word=%#x port=%d\n", word, port_id);
    sendMemRequest(pkt, port_id);
}

uint32_t
SpecializedExecutionUnit::extractCmdWord(const std::vector<uint8_t> &cmd) const
{
    if (cmd.size() < sizeof(uint32_t)) {
        return 0;
    }

    uint32_t word = 0;
    std::memcpy(&word, cmd.data(), sizeof(word));
    return word;
}

SpecializedExecutionUnit::CmdFields
SpecializedExecutionUnit::parseCmdFields(uint32_t word) const
{
    CmdFields fields;
    fields.deviceType = (word >> 28) & 0xF;
    fields.deviceId = (word >> 24) & 0xF;
    fields.opCode = (word >> 16) & 0xFF;
    fields.syncIndicator = (word >> 8) & 0xFF;
    fields.setIndicatorSns = ((word >> 7) & 0x1) != 0;
    fields.setIndicatorSnd = ((word >> 6) & 0x1) != 0;
    return fields;
}

void
SpecializedExecutionUnit::setDebugProcessLatency(Tick latency)
{
    debugProcessLatency = latency;
}

bool
SpecializedExecutionUnit::startBlockingRead(Addr addr, size_t size,
                                             uint8_t *buffer, PortID port_id)
{
    RequestPtr req = std::make_shared<Request>(
        addr, size, Request::Flags(), Request::funcRequestorId);
    PacketPtr pkt = new Packet(req, MemCmd::ReadReq);
    pkt->allocate();
    pkt->dataStatic(buffer);
    sendMemRequest(pkt, port_id);
    return true;
}

bool
SpecializedExecutionUnit::startBlockingWrite(Addr addr, size_t size,
                                              const uint8_t *buffer,
                                              PortID port_id)
{
    RequestPtr req = std::make_shared<Request>(
        addr, size, Request::Flags(), Request::funcRequestorId);
    PacketPtr pkt = new Packet(req, MemCmd::WriteReq);
    pkt->allocate();
    pkt->dataStatic(const_cast<uint8_t *>(buffer));
    sendMemRequest(pkt, port_id);
    return true;
}

bool
SpecializedExecutionUnit::startBlockingRead(Addr addr, size_t size,
                                             uint8_t *buffer)
{
    return startBlockingRead(addr, size, buffer, 0);
}

bool
SpecializedExecutionUnit::startBlockingWrite(Addr addr, size_t size,
                                              const uint8_t *buffer)
{
    return startBlockingWrite(addr, size, buffer, 0);
}

bool
SpecializedExecutionUnit::handleMemResponse(PacketPtr pkt)
{
    auto it = activeMemTxns.find(pkt);
    panic_if(it == activeMemTxns.end(),
             "%s: received response for unknown packet addr=%#x",
             name(), pkt->getAddr());

    const MemTxnContext txn = it->second;
    DPRINTF(SpecializedExecutionUnit,
            "Received memory response for addr=%#x port=%d kind=%d "
            "remaining_before=%zu\n",
            pkt->getAddr(), txn.portId, static_cast<int>(txn.kind),
            activeMemTxns.size());

    switch (txn.kind) {
      case MemTxnContext::Kind::Mvin:
        return handleMvinResponseInternal(activeExecution, pkt);
      case MemTxnContext::Kind::Mvout:
        return handleMvoutResponseInternal(activeExecution, pkt);
      case MemTxnContext::Kind::SyncWrite:
        activeMemTxns.erase(it);
        delete pkt;
        if (activeMemTxns.empty() && !issueCmdBusy && !cmdQueue.empty()) {
            tryScheduleIssue();
        }
        return true;
    }

    panic("%s: unhandled memory transaction kind", name());
}

uint64_t
SpecializedExecutionUnit::queueOccupancy() const
{
    return cmdQueue.size();
}

uint64_t
SpecializedExecutionUnit::completedCmdCount() const
{
    return completedCount;
}

bool
SpecializedExecutionUnit::isIssueBusy() const
{
    return issueCmdBusy;
}

uint64_t
SpecializedExecutionUnit::completedReadRespCount() const
{
    return activeExecution.completedReadRespCount;
}

uint64_t
SpecializedExecutionUnit::completedWriteRespCount() const
{
    return activeExecution.completedWriteRespCount;
}

uint64_t
SpecializedExecutionUnit::completedIterationCount() const
{
    return activeExecution.completedIterations;
}

uint64_t
SpecializedExecutionUnit::prologueCount() const
{
    return activeExecution.prologueCount;
}

uint64_t
SpecializedExecutionUnit::executeCount() const
{
    return activeExecution.executeCount;
}

uint64_t
SpecializedExecutionUnit::epilogueCount() const
{
    return activeExecution.epilogueCount;
}

} // namespace gem5
