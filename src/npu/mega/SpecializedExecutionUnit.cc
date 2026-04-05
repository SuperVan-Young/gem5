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
    : ResponsePort(name),
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
    : RequestPort(name), owner(owner), blockedPacket(nullptr)
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
      maxConcurrentMicroOps(0),
      activeCmd(macroCmdBytes, 0),
      issueEvent([this] { issueOneCommand(); }, name() + ".issueEvent"),
      finishExecutionEvent([this] { finishExecution(); },
                           name() + ".finishExecutionEvent")
{
    panic_if(params.num_mem_side_ports == 0,
             "SpecializedExecutionUnit requires at least one mem_side port");

    stagingBuffer.bytes.resize(macroCmdBytes, 0);
    memSidePorts.reserve(params.num_mem_side_ports);
    loadQueues.resize(params.num_mem_side_ports);
    storeQueues.resize(params.num_mem_side_ports);
    memPortBusy.resize(params.num_mem_side_ports, false);
    for (PortID i = 0; i < params.num_mem_side_ports; ++i) {
        auto port = std::make_unique<MemSidePort>(
            csprintf("%s.mem_side[%d]", params.name, i), this);
        port->portId = i;
        memSidePorts.push_back(std::move(port));
    }

    DPRINTF(SpecializedExecutionUnit,
            "Created SEU: base_addr=%#x cmd_bytes=%u queue_depth=%u "
            "mem_ports=%llu\n",
            baseAddr, macroCmdBytes, cmdQueueDepth,
            static_cast<unsigned long long>(memSidePorts.size()));
}

SpecializedExecutionUnit::~SpecializedExecutionUnit()
{
}

void
SpecializedExecutionUnit::init()
{
    ClockedObject::init();

    if (cpuSidePort.isConnected()) {
        cpuSidePort.sendRangeChange();
    }
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
             "%s: mem_side port index %d out of range (num ports=%llu)",
             name(), idx,
             static_cast<unsigned long long>(memSidePorts.size()));
    return *memSidePorts[idx];
}

const SpecializedExecutionUnit::MemSidePort &
SpecializedExecutionUnit::getMemSidePort(PortID idx) const
{
    panic_if(idx < 0 || static_cast<size_t>(idx) >= memSidePorts.size(),
             "%s: mem_side port index %d out of range (num ports=%llu)",
             name(), idx,
             static_cast<unsigned long long>(memSidePorts.size()));
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
                "Launch rejected: queue full (%llu/%u)\n",
                static_cast<unsigned long long>(cmdQueue.size()),
                cmdQueueDepth);
        return false;
    }
    cmdQueue.push_back(stagingBuffer.bytes);
    DPRINTF(SpecializedExecutionUnit,
            "Launched command, queue size now %llu\n",
            static_cast<unsigned long long>(cmdQueue.size()));
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
            "Issuing command, queue size now %llu\n",
            static_cast<unsigned long long>(cmdQueue.size()));

    startExecuteCommand(activeCmd);
}

void
SpecializedExecutionUnit::startExecuteCommand(const std::vector<uint8_t> &cmd)
{
    activeExecution.cmd = cmd;
    beginActiveCommand(activeExecution);
    prepareIteration(activeExecution, 0);
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
            "Finished command %llu, queue size %llu\n",
            static_cast<unsigned long long>(completedCount),
            static_cast<unsigned long long>(cmdQueue.size()));

    if (!sentCompletionSync && !cmdQueue.empty()) {
        schedule(issueEvent, nextCycle());
    }

    // If we previously blocked a launch due to full queue, retry now
    cpuSidePort.trySendRetry();
}

void
SpecializedExecutionUnit::finishExecution()
{
    panic_if(!activeExecOp.has_value(),
             "%s: exec completion event without active exec op", name());
    const MicroOpContext ctx = *activeExecOp;
    activeExecOp.reset();
    handleExecCompletion(activeExecution, ctx);
    launchExecIfReady();
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
    exec.nextIterationToPrepare = 0;
    exec.nextIterationToRetire = 0;
    exec.completionIssued = false;
    exec.finalizePending = false;

    exec.readMask = readCmdWord(exec.cmd, MultiportReadMaskWord);
    exec.writeMask = readCmdWord(exec.cmd, MultiportWriteMaskWord);
    exec.repetition = readCmdWord(exec.cmd, MultiportRepetitionWord);
    exec.reserved = readCmdWord(exec.cmd, MultiportReservedWord);

    panic_if(!maskFitsPortCount(exec.readMask, memSidePorts.size()),
             "%s: read mask %#x exceeds mem port count %llu",
             name(), exec.readMask,
             static_cast<unsigned long long>(memSidePorts.size()));
    panic_if(!maskFitsPortCount(exec.writeMask, memSidePorts.size()),
             "%s: write mask %#x exceeds mem port count %llu",
             name(), exec.writeMask,
             static_cast<unsigned long long>(memSidePorts.size()));
    if (exec.repetition == 0) {
        exec.repetition = 1;
    }

    for (PortID port = 0; port < static_cast<PortID>(memSidePorts.size());
         ++port) {
        exec.readResults[port] = packWord(defaultInitialSlotValue(port));
        loadQueues[port].clear();
        storeQueues[port].clear();
        memPortBusy[port] = false;
    }
    execQueue.clear();
    activeExecOp.reset();
    iterationStates.clear();
    maxConcurrentMicroOps = 0;

    onCommandBegin(exec);
}

void
SpecializedExecutionUnit::prepareIteration(ActiveExecution &exec,
                                           uint64_t iteration)
{
    if (iteration >= exec.repetition) {
        return;
    }
    auto [iter_it, inserted] =
        iterationStates.emplace(iteration, IterationState{});
    IterationState &state = iter_it->second;
    if (state.prepared) {
        return;
    }

    exec.phase = Phase::Prologue;
    exec.iteration = iteration;
    exec.readPorts.clear();
    exec.writePorts.clear();
    exec.writeResults.clear();
    exec.prologueCount++;
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
    state.prepared = true;
    prologue(exec);
    exec.nextIterationToPrepare = std::max(exec.nextIterationToPrepare,
                                           iteration + 1);
    issueLoadRequests(exec, iteration);
}

void
SpecializedExecutionUnit::issueLoadRequests(ActiveExecution &exec,
                                            uint64_t iteration)
{
    exec.iteration = iteration;
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
        handleIterationLoadsReady(exec, iteration);
        return;
    }
    IterationState &state = iterationStates.at(iteration);
    state.pendingLoads = reqs.size();
    for (auto &req_desc : reqs) {
        req_desc.kind = MemTxnContext::Kind::Mvin;
        req_desc.iteration = iteration;
        req_desc.token = nextMemTxnToken++;
        enqueueLoadRequest(req_desc);
    }
}

void
SpecializedExecutionUnit::enqueueLoadRequest(const MemRequestDesc &req_desc)
{
    panic_if(req_desc.portId < 0 ||
             static_cast<size_t>(req_desc.portId) >= loadQueues.size(),
             "%s: invalid load queue port %d", name(), req_desc.portId);
    loadQueues[req_desc.portId].push_back({req_desc});
    pumpMemPort(req_desc.portId);
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
    exec.iteration = txn.iteration;
    exec.readResults[txn.portId] =
        std::vector<uint8_t>(data, data + pkt->getSize());
    exec.completedReadRespCount++;
    onMvinResponse(exec, txn, pkt);
    onMicroOpComplete(exec, MicroOpContext{
        MicroOpContext::Kind::Load, txn.portId, txn.iteration,
        txn.token, txn.addr, txn.size, 0}, pkt);

    activeMemTxns.erase(it);
    memPortBusy[txn.portId] = false;
    delete pkt;
    pumpMemPort(txn.portId);

    IterationState &state = iterationStates.at(txn.iteration);
    panic_if(state.pendingLoads == 0,
             "%s: unexpected mvin completion for iteration %llu", name(),
             static_cast<unsigned long long>(txn.iteration));
    state.pendingLoads--;
    if (state.pendingLoads == 0) {
        handleIterationLoadsReady(exec, txn.iteration);
    }
    return true;
}

void
SpecializedExecutionUnit::handleIterationLoadsReady(ActiveExecution &exec,
                                                    uint64_t iteration)
{
    exec.iteration = iteration;
    IterationState &state = iterationStates.at(iteration);
    if (state.execQueued) {
        return;
    }
    state.execQueued = true;
    exec.executeCount++;
    const Tick execLatency = execute(exec);
    enqueueExecOp(iteration, execLatency);
    if (iteration + 1 < exec.repetition) {
        prepareIteration(exec, iteration + 1);
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
    exec.iteration = txn.iteration;
    exec.completedWriteRespCount++;
    if (auto write_it = exec.writeResults.find(txn.portId);
        write_it != exec.writeResults.end()) {
        exec.readResults[txn.portId] = write_it->second;
    }
    onMvoutResponse(exec, txn, pkt);
    onMicroOpComplete(exec, MicroOpContext{
        MicroOpContext::Kind::Store, txn.portId, txn.iteration,
        txn.token, txn.addr, txn.size, 0}, pkt);

    activeMemTxns.erase(it);
    memPortBusy[txn.portId] = false;
    delete pkt;
    pumpMemPort(txn.portId);

    IterationState &state = iterationStates.at(txn.iteration);
    panic_if(state.pendingStores == 0,
             "%s: unexpected mvout completion for iteration %llu", name(),
             static_cast<unsigned long long>(txn.iteration));
    state.pendingStores--;
    if (state.pendingStores == 0) {
        runEpiloguePhase(exec, txn.iteration);
    }
    return true;
}

void
SpecializedExecutionUnit::enqueueStoreRequest(const MemRequestDesc &req_desc)
{
    panic_if(req_desc.portId < 0 ||
             static_cast<size_t>(req_desc.portId) >= storeQueues.size(),
             "%s: invalid store queue port %d", name(), req_desc.portId);
    storeQueues[req_desc.portId].push_back({req_desc});
    pumpMemPort(req_desc.portId);
}

void
SpecializedExecutionUnit::enqueueExecOp(uint64_t iteration, Tick latency)
{
    execQueue.push_back({MicroOpContext{
        MicroOpContext::Kind::Exec,
        InvalidPortID,
        iteration,
        nextMemTxnToken++,
        0,
        0,
        latency,
    }});
    launchExecIfReady();
}

void
SpecializedExecutionUnit::pumpMemPort(PortID port_id)
{
    if (memPortBusy[port_id]) {
        return;
    }

    std::deque<QueuedMemOp> *queue = nullptr;
    if (!loadQueues[port_id].empty()) {
        queue = &loadQueues[port_id];
    } else if (!storeQueues[port_id].empty()) {
        queue = &storeQueues[port_id];
    }

    if (queue == nullptr) {
        return;
    }

    const MemRequestDesc req_desc = queue->front().desc;
    queue->pop_front();

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
    memPortBusy[port_id] = true;
    getMemSidePort(port_id).sendPacket(pkt);
    updateConcurrentMicroOps();
}

void
SpecializedExecutionUnit::launchExecIfReady()
{
    if (activeExecOp.has_value() || execQueue.empty()) {
        return;
    }

    activeExecOp = execQueue.front().ctx;
    execQueue.pop_front();
    updateConcurrentMicroOps();
    schedule(finishExecutionEvent,
             curTick() + std::max<Tick>(1, activeExecOp->latency));
}

void
SpecializedExecutionUnit::handleExecCompletion(ActiveExecution &exec,
                                               const MicroOpContext &ctx)
{
    exec.iteration = ctx.iteration;
    IterationState &state = iterationStates.at(ctx.iteration);
    state.execCompleted = true;
    onMicroOpComplete(exec, ctx, nullptr);
    issueStoreRequests(exec, ctx.iteration);
}

void
SpecializedExecutionUnit::issueStoreRequests(ActiveExecution &exec,
                                             uint64_t iteration)
{
    exec.iteration = iteration;
    exec.phase = Phase::Completing;
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
                    ((static_cast<uint32_t>(iteration) + 1U) * 0x10U) +
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

    IterationState &state = iterationStates.at(iteration);
    state.pendingStores = reqs.size();
    if (reqs.empty()) {
        runEpiloguePhase(exec, iteration);
        return;
    }

    for (auto &req_desc : reqs) {
        req_desc.kind = MemTxnContext::Kind::Mvout;
        req_desc.iteration = iteration;
        req_desc.token = nextMemTxnToken++;
        enqueueStoreRequest(req_desc);
    }
}

void
SpecializedExecutionUnit::retireCompletedIterations(ActiveExecution &exec)
{
    while (exec.nextIterationToRetire < exec.repetition) {
        auto it = iterationStates.find(exec.nextIterationToRetire);
        if (it == iterationStates.end() || !it->second.epilogueDone) {
            break;
        }

        exec.completedIterations++;
        exec.nextIterationToRetire++;
    }

    if (exec.completedIterations >= exec.repetition &&
        activeMemTxns.empty() && execQueue.empty() &&
        !activeExecOp.has_value()) {
        finalizeActiveCommand(exec);
    }
}

void
SpecializedExecutionUnit::runEpiloguePhase(ActiveExecution &exec,
                                           uint64_t iteration)
{
    exec.phase = Phase::Epilogue;
    exec.iteration = iteration;
    IterationState &state = iterationStates.at(iteration);
    if (state.epilogueDone) {
        return;
    }

    exec.epilogueCount++;
    epilogue(exec);
    state.epilogueDone = true;

    retireCompletedIterations(exec);
}

void
SpecializedExecutionUnit::finalizeActiveCommand(ActiveExecution &exec)
{
    if (exec.completionIssued) {
        return;
    }

    ActiveExecution exit_view = exec;
    exit_view.completedIterations = exec.repetition;
    panic_if(!shouldExit(exit_view),
             "%s: command reached finalizeActiveCommand before shouldExit",
             name());

    exec.phase = Phase::Completing;
    exec.completionIssued = true;
    completeActiveCommand();
    exec.phase = Phase::Idle;
}

void
SpecializedExecutionUnit::updateConcurrentMicroOps()
{
    const uint64_t active_ops =
        activeMemTxns.size() + (activeExecOp.has_value() ? 1U : 0U);
    maxConcurrentMicroOps = std::max(maxConcurrentMicroOps, active_ops);
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
            "sendMemRequest: port=%d kind=%d addr=%#x size=%u inflight=%llu\n",
            port_id, static_cast<int>(txn.kind), txn.addr, txn.size,
            static_cast<unsigned long long>(activeMemTxns.size()));

    getMemSidePort(port_id).sendPacket(pkt);
    updateConcurrentMicroOps();
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
            "remaining_before=%llu\n",
            pkt->getAddr(), txn.portId, static_cast<int>(txn.kind),
            static_cast<unsigned long long>(activeMemTxns.size()));

    switch (txn.kind) {
      case MemTxnContext::Kind::Mvin:
        return handleMvinResponseInternal(activeExecution, pkt);
      case MemTxnContext::Kind::Mvout:
        return handleMvoutResponseInternal(activeExecution, pkt);
      case MemTxnContext::Kind::SyncWrite:
        onMicroOpComplete(activeExecution, MicroOpContext{
            MicroOpContext::Kind::SyncWrite, txn.portId, txn.iteration,
            txn.token, txn.addr, txn.size, 0}, pkt);
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

uint64_t
SpecializedExecutionUnit::maxActiveMicroOps() const
{
    return maxConcurrentMicroOps;
}

} // namespace gem5
