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

#include "npu/SpecializedExecutionUnit.hh"

#include <algorithm>
#include <cstring>
#include <sstream>

#include "base/trace.hh"
#include "debug/NPULaunchProfile.hh"
#include "debug/NPUProfile.hh"
#include "debug/SpecializedExecutionUnit.hh"

namespace gem5
{

namespace
{

constexpr Addr SyncIndicatorBase = 0x71000000;
constexpr Addr DefaultSpmBase = 0x60000000;
constexpr Addr DefaultSpmSlotStride = 0x40;
constexpr size_t QueueSelectWord = 1;
constexpr size_t UopCountWord = 2;
constexpr size_t Aux0Word = 3;
constexpr size_t Aux1Word = 4;

PacketPtr
buildPacketFromRequest(const RequestPtr &req,
                       MemCmd cmd,
                       const std::vector<uint8_t> *data)
{
    PacketPtr pkt = new Packet(req, cmd);
    pkt->allocate();
    if (cmd == MemCmd::WriteReq && data && !data->empty()) {
        pkt->setData(data->data());
    }
    return pkt;
}

std::vector<uint8_t>
packWord(uint32_t value)
{
    std::vector<uint8_t> bytes(sizeof(value), 0);
    std::memcpy(bytes.data(), &value, sizeof(value));
    return bytes;
}

void
appendLaunchFieldsJson(std::ostream &os,
                       const SpecializedExecutionUnit::CmdFields &fields)
{
    os << "\"device_type\":" << static_cast<uint32_t>(fields.deviceType);
    os << ",\"device_id\":" << static_cast<uint32_t>(fields.deviceId);
    os << ",\"opcode\":" << static_cast<uint32_t>(fields.opCode);
    os << ",\"sync_indicator\":"
       << static_cast<uint32_t>(fields.syncIndicator);
}

void
emitLaunchProfile(const std::string &owner_name, const char *stage, Tick tick,
                  uint64_t macro_id,
                  const SpecializedExecutionUnit::MacroCmdContext &macro_cmd,
                  uint64_t dispatch_depth)
{
    std::ostringstream os;
    os << '{';
    os << "\"event\":\"" << stage << "\"";
    os << ",\"tick\":" << tick;
    os << ",\"owner\":\"" << owner_name << "\"";
    os << ",\"macro_id\":" << macro_id;
    os << ",\"issue_queue\":" << macro_cmd.targetIssueQueueId;
    os << ",\"dispatch_depth\":" << dispatch_depth;
    os << ',';
    appendLaunchFieldsJson(os, macro_cmd.fields);
    os << '}';
    DPRINTF(NPULaunchProfile, "NPU_LAUNCH_PROFILE %s\n", os.str().c_str());
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
    owner->trySendCompletionSyncWords();
}

SpecializedExecutionUnit::SpecializedExecutionUnit(
    const SpecializedExecutionUnitParams &params)
    : ClockedObject(params),
      cpuSidePort(params.name + ".cpu_side", this),
      macroCmdBytes(params.macro_cmd_bytes),
      cmdQueueDepth(params.cmd_queue_depth),
      baseAddr(params.base_addr),
      syncEnqueueOnDataWrite(params.sync_enqueue_on_data_write),
      memPortOutstandingLimit(params.mem_port_outstanding_limit),
      debugProcessLatency(params.debug_process_latency),
      issueEvent([this] { issueOneCommand(); }, name() + ".issueEvent")
{
    panic_if(params.num_mem_side_ports == 0,
             "SpecializedExecutionUnit requires at least one mem_side port");
    panic_if(memPortOutstandingLimit == 0,
             "%s: mem_port_outstanding_limit must be non-zero", name());

    stagingBuffer.bytes.resize(macroCmdBytes, 0);
    memSidePorts.reserve(params.num_mem_side_ports);
    for (PortID i = 0; i < params.num_mem_side_ports; ++i) {
        auto port = std::make_unique<MemSidePort>(
            csprintf("%s.mem_side[%d]", params.name, i), this);
        port->portId = i;
        memSidePorts.push_back(std::move(port));
    }
    memPortOutstanding.resize(params.num_mem_side_ports, 0);
    maxMemPortOutstandingValues.resize(params.num_mem_side_ports, 0);
    issueQueues = buildIssueQueues();

    DPRINTF(SpecializedExecutionUnit,
            "Created SEU: base_addr=%#x cmd_bytes=%u queue_depth=%u "
            "mem_ports=%llu issue_queues=%llu outstanding_limit=%u\n",
            baseAddr, macroCmdBytes, cmdQueueDepth,
            static_cast<unsigned long long>(memSidePorts.size()),
            static_cast<unsigned long long>(issueQueues.size()),
            memPortOutstandingLimit);
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
SpecializedExecutionUnit::writeDataBytes(Addr offset,
                                         const uint8_t *src,
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
    const size_t size = pkt->getSize();
    if (!validMmioOffset(offset, size)) {
        return false;
    }
    return writeDataBytes(offset, pkt->getConstPtr<uint8_t>(), size);
}

bool
SpecializedExecutionUnit::canLaunchCmd() const
{
    return queueOccupancy() < cmdQueueDepth;
}

bool
SpecializedExecutionUnit::launchStagedCmd()
{
    if (!canLaunchCmd()) {
        DPRINTF(SpecializedExecutionUnit,
                "Launch rejected: queue full occupancy=%llu depth=%u\n",
                static_cast<unsigned long long>(queueOccupancy()),
                cmdQueueDepth);
        return false;
    }

    MacroCmdContext macro_cmd;
    macro_cmd.macroCmdId = nextMacroCmdId++;
    macro_cmd.cmd = stagingBuffer.bytes;
    macro_cmd.fields = parseCmdFields(extractCmdWord(macro_cmd.cmd));
    macro_cmd.kind = classifyMacroCmd(macro_cmd.cmd);
    macro_cmd.targetIssueQueueId =
        classifyIssueQueue(macro_cmd.cmd, macro_cmd.kind);
    panic_if(!issueQueueExists(macro_cmd.targetIssueQueueId),
             "%s: macro command chose missing issue queue %u",
             name(), macro_cmd.targetIssueQueueId);
    if (const auto &queue = getIssueQueue(macro_cmd.targetIssueQueueId);
        queue.mappedMemPortId.has_value()) {
        macro_cmd.boundMemPortId = queue.mappedMemPortId;
    }

    macroCmdContexts.emplace(macro_cmd.macroCmdId, std::move(macro_cmd));
    dispatchQueue.push_back(nextMacroCmdId - 1);
    emitLaunchProfile(name(), "seu_accept", curTick(), nextMacroCmdId - 1,
                      macroCmdContexts.find(nextMacroCmdId - 1)->second,
                      dispatchQueue.size());

    DPRINTF(SpecializedExecutionUnit,
            "Launched macro command id=%llu dispatch=%llu occupancy=%llu\n",
            static_cast<unsigned long long>(nextMacroCmdId - 1),
            static_cast<unsigned long long>(dispatchQueue.size()),
            static_cast<unsigned long long>(queueOccupancy()));

    tryScheduleIssue();
    return true;
}

bool
SpecializedExecutionUnit::handleRequest(PacketPtr pkt)
{
    if (!pkt->isWrite()) {
        panic("SpecializedExecutionUnit only accepts write requests");
    }

    const Addr addr = pkt->getAddr();
    const Addr offset = addr - baseAddr;
    if (!validMmioOffset(offset, pkt->getSize())) {
        panic("MMIO write out of bounds: addr=%#x offset=%#x size=%u",
              addr, offset, pkt->getSize());
    }

    DPRINTF(SpecializedExecutionUnit,
            "MMIO write addr=%#x offset=%#x size=%u\n",
            addr, offset, pkt->getSize());

    bool success = true;
    if (offset < macroCmdBytes) {
        writeDataChunk(offset, pkt);
        if (syncEnqueueOnDataWrite && offset == 0) {
            success = launchStagedCmd();
        }
    } else {
        success = launchStagedCmd();
    }

    return success;
}

void
SpecializedExecutionUnit::tryScheduleIssue()
{
    if (!issueEvent.scheduled() && hasSchedulableWork()) {
        schedule(issueEvent, nextCycle());
    }
}

bool
SpecializedExecutionUnit::hasSchedulableWork() const
{
    if (!dispatchQueue.empty() || !pendingEpilogueCmdIds.empty()) {
        return true;
    }

    for (const auto &queue : issueQueues) {
        if (queue.activeMacroCmdId.has_value() ||
            !queue.waitingMacroCmdIds.empty()) {
            return true;
        }
    }

    return false;
}

void
SpecializedExecutionUnit::issueOneCommand()
{
    runPendingEpilogues();
    dispatchCommands();
    activateIssueQueues();
    issueReadyUops();

    if (hasSchedulableWork()) {
        tryScheduleIssue();
    }
}

void
SpecializedExecutionUnit::dispatchCommands()
{
    while (!dispatchQueue.empty()) {
        const uint64_t macro_id = dispatchQueue.front();
        dispatchQueue.pop_front();
        auto it = macroCmdContexts.find(macro_id);
        if (it == macroCmdContexts.end()) {
            continue;
        }

        auto &macro_cmd = it->second;
        auto &queue = getIssueQueue(macro_cmd.targetIssueQueueId);
        queue.waitingMacroCmdIds.push_back(macro_id);

        DPRINTF(SpecializedExecutionUnit,
                "Dispatch macro=%llu to issue_queue=%u waiters=%llu\n",
                static_cast<unsigned long long>(macro_id),
                macro_cmd.targetIssueQueueId,
                static_cast<unsigned long long>(
                    queue.waitingMacroCmdIds.size()));
    }
}

void
SpecializedExecutionUnit::activateIssueQueues()
{
    for (auto &queue : issueQueues) {
        if (queue.activeMacroCmdId.has_value() ||
            queue.waitingMacroCmdIds.empty()) {
            continue;
        }

        const uint64_t macro_id = queue.waitingMacroCmdIds.front();
        auto it = macroCmdContexts.find(macro_id);
        panic_if(it == macroCmdContexts.end(),
                 "%s: missing macro command %llu during activation",
                 name(), static_cast<unsigned long long>(macro_id));
        if (!canActivateMacroCmd(it->second)) {
            continue;
        }

        queue.waitingMacroCmdIds.pop_front();
        queue.activeMacroCmdId = macro_id;
        runMacroCmdPrologue(it->second);
    }
}

void
SpecializedExecutionUnit::runMacroCmdPrologue(MacroCmdContext &macroCmd)
{
    panic_if(!macroCmd.uopQueue.empty(),
             "%s: prologue requires empty uop queue for macro %llu",
             name(), static_cast<unsigned long long>(macroCmd.macroCmdId));

    macroCmd.phase = Phase::Prologue;
    prologueCountValue++;
    onMacroCmdBegin(macroCmd);
    emitLaunchProfile(name(), "seu_begin", curTick(), macroCmd.macroCmdId,
                      macroCmd, dispatchQueue.size());
    emitProfileBegin(macroCmd);
    buildUops(macroCmd);

    if (macroCmd.phase == Phase::EpiloguePending) {
        markEpiloguePending(macroCmd);
        return;
    }

    macroCmd.phase = Phase::Active;
}

void
SpecializedExecutionUnit::issueReadyUops()
{
    for (auto &queue : issueQueues) {
        if (!queue.activeMacroCmdId.has_value()) {
            continue;
        }

        auto it = macroCmdContexts.find(*queue.activeMacroCmdId);
        if (it == macroCmdContexts.end()) {
            continue;
        }
        auto &macro_cmd = it->second;

        if (macro_cmd.phase != Phase::Active || macro_cmd.uopQueue.empty()) {
            continue;
        }

        while (macro_cmd.phase == Phase::Active &&
               !macro_cmd.uopQueue.empty()) {
            const auto &front = macro_cmd.uopQueue.front();
            if (front.kind == MicroOpContext::Kind::Load ||
                front.kind == MicroOpContext::Kind::Store) {
                if (!canIssueMemUop(front, macro_cmd)) {
                    break;
                }
            } else if (macro_cmd.waitingCallback ||
                       macro_cmd.outstandingMemUops != 0) {
                break;
            }

            MicroOpContext uop = std::move(macro_cmd.uopQueue.front());
            macro_cmd.uopQueue.pop_front();
            const auto kind = uop.kind;
            issueOneUop(macro_cmd, std::move(uop));

            if (kind != MicroOpContext::Kind::Load &&
                kind != MicroOpContext::Kind::Store) {
                break;
            }
        }
    }
}

void
SpecializedExecutionUnit::issueOneUop(MacroCmdContext &macroCmd,
                                      MicroOpContext uop)
{
    panic_if(macroCmd.phase != Phase::Active,
             "%s: issueOneUop requires Active phase for macro %llu",
             name(), static_cast<unsigned long long>(macroCmd.macroCmdId));

    switch (uop.kind) {
      case MicroOpContext::Kind::Load:
      case MicroOpContext::Kind::Store:
        issueMemUop(macroCmd, std::move(uop));
        break;
      case MicroOpContext::Kind::Exec:
        macroCmd.waitingCallback = true;
        issueExecUop(macroCmd, std::move(uop));
        break;
      case MicroOpContext::Kind::SyncWrite:
        panic("%s: sync-write uops are not issued from issue queues", name());
    }
}

void
SpecializedExecutionUnit::issueMemUop(MacroCmdContext &macroCmd,
                                      MicroOpContext uop)
{
    uop.profileId = nextUopProfileId++;
    if (uop.kind == MicroOpContext::Kind::Load) {
        macroCmd.issuedLoadUops++;
    } else {
        macroCmd.issuedStoreUops++;
    }
    recordUopIssue(macroCmd, uop);

    const PortID port_id =
        uop.portId == InvalidPortID ? mappedMemPort(macroCmd) : uop.portId;

    MemTxnContext txn;
    txn.kind = uop.kind == MicroOpContext::Kind::Load ?
        MemTxnContext::Kind::Load : MemTxnContext::Kind::Store;
    txn.macroCmdId = macroCmd.macroCmdId;
    txn.ownerIssueQueueId = macroCmd.targetIssueQueueId;
    txn.portId = port_id;
    txn.token = uop.token;
    txn.profileId = uop.profileId;
    txn.addr = uop.addr;
    txn.size = uop.size;

    ++macroCmd.outstandingMemUops;
    ++memPortOutstanding[port_id];
    maxMemPortOutstandingValues[port_id] =
        std::max<uint64_t>(maxMemPortOutstandingValues[port_id],
                           memPortOutstanding[port_id]);

    sendTrackedPacket(txn, uop.kind == MicroOpContext::Kind::Store ?
        &uop.data : nullptr);
}

void
SpecializedExecutionUnit::issueExecUop(MacroCmdContext &macroCmd,
                                       MicroOpContext uop)
{
    uop.profileId = nextUopProfileId++;
    const uint32_t queue_id = macroCmd.targetIssueQueueId;
    panic_if(activeExecUops.find(queue_id) != activeExecUops.end(),
             "%s: issue queue %u already has an active exec uop",
             name(), queue_id);

    executeCountValue++;
    macroCmd.issuedExecUops++;
    macroCmd.execActiveTicks += std::max<Tick>(1, uop.latency);
    recordUopIssue(macroCmd, uop);
    activeExecUops.emplace(queue_id, uop);
    auto event = std::make_unique<EventFunctionWrapper>(
        [this, queue_id] { finishExecution(queue_id); },
        csprintf("%s.execComplete[%u]", name(), queue_id));
    schedule(*event, curTick() + std::max<Tick>(1, uop.latency));
    execCompletionEvents.emplace(queue_id, std::move(event));
    updateConcurrentMicroOps();
}

void
SpecializedExecutionUnit::finishExecution(uint32_t issueQueueId)
{
    auto uop_it = activeExecUops.find(issueQueueId);
    panic_if(uop_it == activeExecUops.end(),
             "%s: exec completion without active exec uop on queue %u",
             name(), issueQueueId);
    const MicroOpContext uop = uop_it->second;
    activeExecUops.erase(uop_it);
    execCompletionEvents.erase(issueQueueId);

    auto macro_it = macroCmdContexts.find(uop.macroCmdId);
    panic_if(macro_it == macroCmdContexts.end(),
             "%s: missing macro command %llu for exec completion",
             name(), static_cast<unsigned long long>(uop.macroCmdId));

    auto &macro_cmd = macro_it->second;
    macro_cmd.waitingCallback = false;
    recordUopComplete(macro_cmd, uop.profileId);
    onExecUopComplete(macro_cmd, uop);

    updateConcurrentMicroOps();
    tryScheduleIssue();
}

void
SpecializedExecutionUnit::runPendingEpilogues()
{
    const size_t count = pendingEpilogueCmdIds.size();
    for (size_t i = 0; i < count; ++i) {
        const uint64_t macro_id = pendingEpilogueCmdIds.front();
        pendingEpilogueCmdIds.pop_front();

        auto it = macroCmdContexts.find(macro_id);
        if (it == macroCmdContexts.end()) {
            continue;
        }
        auto &macro_cmd = it->second;
        macro_cmd.epilogueQueued = false;
        if (macro_cmd.phase != Phase::EpiloguePending) {
            continue;
        }
        runMacroCmdEpilogue(macro_cmd);
    }
}

void
SpecializedExecutionUnit::runMacroCmdEpilogue(MacroCmdContext &macroCmd)
{
    panic_if(!macroCmd.uopQueue.empty(),
             "%s: epilogue requires empty uop queue for macro %llu",
             name(), static_cast<unsigned long long>(macroCmd.macroCmdId));
    panic_if(macroCmd.waitingCallback,
             "%s: epilogue requires no outstanding callback for macro %llu",
             name(), static_cast<unsigned long long>(macroCmd.macroCmdId));

    macroCmd.phase = Phase::Epilogue;
    epilogueCountValue++;
    emitProfileEnd(macroCmd);
    onMacroCmdEnd(macroCmd);
    finishMacroCmd(macroCmd);
}

void
SpecializedExecutionUnit::finishMacroCmd(MacroCmdContext &macroCmd)
{
    if (macroCmd.completionIssued) {
        return;
    }

    macroCmd.completionIssued = true;
    macroCmd.phase = Phase::Done;
    completedCount++;
    completedIterationValue++;

    const uint64_t macro_id = macroCmd.macroCmdId;
    const uint32_t queue_id = macroCmd.targetIssueQueueId;
    uint32_t sync_word = 0;
    if (buildCompletionSyncWord(macroCmd.cmd, sync_word)) {
        sendCompletionSyncWord(sync_word);
    }

    DPRINTF(SpecializedExecutionUnit,
            "Finished macro=%llu queue=%u completed=%llu occupancy=%llu\n",
            static_cast<unsigned long long>(macro_id), queue_id,
            static_cast<unsigned long long>(completedCount),
            static_cast<unsigned long long>(queueOccupancy()));

    releaseIssueQueueOwner(queue_id, macro_id);
    macroCmdContexts.erase(macro_id);
    cpuSidePort.trySendRetry();
}

void
SpecializedExecutionUnit::releaseIssueQueueOwner(uint32_t issueQueueId,
                                                 uint64_t macroCmdId)
{
    auto &queue = getIssueQueue(issueQueueId);
    if (queue.activeMacroCmdId.has_value() &&
        *queue.activeMacroCmdId == macroCmdId) {
        queue.activeMacroCmdId.reset();
    }
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
            "Received mem response macro=%llu queue=%u port=%d kind=%d\n",
            static_cast<unsigned long long>(txn.macroCmdId),
            txn.ownerIssueQueueId, txn.portId, static_cast<int>(txn.kind));

    activeMemTxns.erase(it);

    if (txn.kind == MemTxnContext::Kind::SyncWrite) {
        delete pkt;
        updateConcurrentMicroOps();
        trySendCompletionSyncWords();
        tryScheduleIssue();
        return true;
    }

    auto macro_it = macroCmdContexts.find(txn.macroCmdId);
    panic_if(macro_it == macroCmdContexts.end(),
             "%s: missing macro command %llu for mem response",
             name(), static_cast<unsigned long long>(txn.macroCmdId));

    auto &macro_cmd = macro_it->second;
    panic_if(macro_cmd.outstandingMemUops == 0,
             "%s: macro %llu memory response underflow",
             name(), static_cast<unsigned long long>(txn.macroCmdId));
    panic_if(txn.portId < 0 ||
             static_cast<size_t>(txn.portId) >= memPortOutstanding.size(),
             "%s: invalid memory response port %d", name(), txn.portId);
    panic_if(memPortOutstanding[txn.portId] == 0,
             "%s: memory port %d outstanding underflow", name(), txn.portId);

    --macro_cmd.outstandingMemUops;
    --memPortOutstanding[txn.portId];
    if (txn.kind == MemTxnContext::Kind::Load) {
        completedReadCount++;
        macro_cmd.completedLoadUops++;
    } else {
        completedWriteCount++;
        macro_cmd.completedStoreUops++;
    }
    recordUopComplete(macro_cmd, txn.profileId);
    onMemUopComplete(macro_cmd, txn, pkt);

    delete pkt;
    updateConcurrentMicroOps();
    trySendCompletionSyncWords();
    tryScheduleIssue();
    return true;
}

void
SpecializedExecutionUnit::sendTrackedPacket(const MemTxnContext &txn,
                                            const std::vector<uint8_t> *data)
{
    const MemCmd cmd = txn.kind == MemTxnContext::Kind::Load ?
        MemCmd::ReadReq : MemCmd::WriteReq;
    RequestPtr req = std::make_shared<Request>(
        txn.addr, txn.size, Request::Flags(), Request::funcRequestorId);
    PacketPtr pkt = buildPacketFromRequest(req, cmd, data);

    MemTxnContext tracked = txn;
    tracked.pkt = pkt;
    auto [it, inserted] = activeMemTxns.emplace(pkt, tracked);
    panic_if(!inserted, "%s: duplicate in-flight packet registration", name());

    DPRINTF(SpecializedExecutionUnit,
            "sendTrackedPacket macro=%llu queue=%u port=%d kind=%d "
            "addr=%#x size=%u\n",
            static_cast<unsigned long long>(txn.macroCmdId),
            txn.ownerIssueQueueId, txn.portId, static_cast<int>(txn.kind),
            txn.addr, static_cast<unsigned>(txn.size));

    getMemSidePort(txn.portId).sendPacket(pkt);
    updateConcurrentMicroOps();
}

bool
SpecializedExecutionUnit::canIssueMemUop(
    const MicroOpContext &uop, const MacroCmdContext &macroCmd) const
{
    const PortID port_id =
        uop.portId == InvalidPortID ? mappedMemPort(macroCmd) : uop.portId;
    panic_if(port_id < 0 ||
             static_cast<size_t>(port_id) >= memPortOutstanding.size(),
             "%s: invalid mem port %d", name(), port_id);
    return memPortOutstanding[port_id] < memPortOutstandingLimit &&
           !getMemSidePort(port_id).isBlocked();
}

void
SpecializedExecutionUnit::updateConcurrentMicroOps()
{
    const uint64_t active_ops =
        activeMemTxns.size() + activeExecUops.size();
    maxConcurrentMicroOpsValue =
        std::max(maxConcurrentMicroOpsValue, active_ops);
}

SpecializedExecutionUnit::MacroCmdKind
SpecializedExecutionUnit::classifyMacroCmd(
    const std::vector<uint8_t> &cmd) const
{
    switch (parseCmdFields(extractCmdWord(cmd)).opCode) {
      case 0:
        return MacroCmdKind::Exec;
      case 1:
        return MacroCmdKind::Load;
      case 2:
        return MacroCmdKind::Store;
      default:
        panic("%s: unsupported base SEU opcode %#x", name(),
              parseCmdFields(extractCmdWord(cmd)).opCode);
    }
}

uint32_t
SpecializedExecutionUnit::classifyIssueQueue(const std::vector<uint8_t> &cmd,
                                             MacroCmdKind kind) const
{
    const uint32_t queue_select = readCmdWord(cmd, QueueSelectWord);
    if (kind == MacroCmdKind::Exec) {
        return 0;
    }

    panic_if(memSidePorts.empty(), "%s: no mem-side ports configured", name());
    return 1 + (queue_select % memSidePorts.size());
}

std::vector<SpecializedExecutionUnit::IssueQueueState>
SpecializedExecutionUnit::buildIssueQueues() const
{
    std::vector<IssueQueueState> queues;
    queues.push_back({0, IssueQueueKind::Exec, {}, {}, {}});
    for (PortID port = 0; port < static_cast<PortID>(memSidePorts.size());
         ++port) {
        queues.push_back({static_cast<uint32_t>(port) + 1,
                          IssueQueueKind::Mem, {}, {}, port});
    }
    return queues;
}

bool
SpecializedExecutionUnit::canActivateMacroCmd(
    const MacroCmdContext &macroCmd) const
{
    (void)macroCmd;
    return true;
}

void
SpecializedExecutionUnit::onMacroCmdBegin(MacroCmdContext &macroCmd)
{
    (void)macroCmd;
}

void
SpecializedExecutionUnit::buildUops(MacroCmdContext &macroCmd)
{
    const uint32_t uop_count =
        std::max<uint32_t>(1, readCmdWord(macroCmd.cmd, UopCountWord));
    const uint32_t aux0 = readCmdWord(macroCmd.cmd, Aux0Word);
    const uint32_t aux1 = readCmdWord(macroCmd.cmd, Aux1Word);

    switch (macroCmd.kind) {
      case MacroCmdKind::Exec: {
        const Tick per_uop_latency = aux0 == 0 ?
            debugProcessLatency :
            clockPeriod() * std::max<uint32_t>(1, aux0);
        for (uint32_t i = 0; i < uop_count; ++i) {
            appendExecUop(macroCmd, per_uop_latency);
        }
        break;
      }
      case MacroCmdKind::Load:
      case MacroCmdKind::Store: {
        const PortID port_id = mappedMemPort(macroCmd);
        const Addr base_addr =
            DefaultSpmBase +
            (static_cast<Addr>(port_id) * DefaultSpmSlotStride);
        const uint32_t stride = aux1 == 0 ? sizeof(uint32_t) : aux1;
        for (uint32_t i = 0; i < uop_count; ++i) {
            const Addr addr =
                base_addr + aux0 + (static_cast<Addr>(i) * stride);
            if (macroCmd.kind == MacroCmdKind::Load) {
                appendLoadUop(macroCmd, addr, sizeof(uint32_t));
            } else {
                const uint32_t value =
                    0xAC000000U + (static_cast<uint32_t>(port_id) << 12) +
                    (static_cast<uint32_t>(macroCmd.macroCmdId) << 4) + i;
                appendStoreUop(macroCmd, addr, sizeof(uint32_t),
                               packWord(value));
            }
        }
        break;
      }
    }

    if (macroCmd.uopQueue.empty()) {
        markEpiloguePending(macroCmd);
    }
}

void
SpecializedExecutionUnit::onMemUopComplete(MacroCmdContext &macroCmd,
                                           const MemTxnContext &txn,
                                           PacketPtr pkt)
{
    (void)txn;
    (void)pkt;
    if (macroCmd.uopQueue.empty()) {
        markEpiloguePending(macroCmd);
    }
}

void
SpecializedExecutionUnit::onExecUopComplete(MacroCmdContext &macroCmd,
                                            const MicroOpContext &uop)
{
    (void)uop;
    if (macroCmd.uopQueue.empty()) {
        markEpiloguePending(macroCmd);
    }
}

void
SpecializedExecutionUnit::onMacroCmdEnd(MacroCmdContext &macroCmd)
{
    (void)macroCmd;
}

const char *
SpecializedExecutionUnit::profileSeuType() const
{
    return "SEU";
}

void
SpecializedExecutionUnit::appendProfileDetailsJson(
    const MacroCmdContext &macroCmd, std::ostream &os) const
{
    os << "\"uop_count_hint\":" <<
        std::max<uint32_t>(1, readCmdWord(macroCmd.cmd, UopCountWord)) <<
        ",\"queue_select\":" << readCmdWord(macroCmd.cmd, QueueSelectWord) <<
        ",\"aux0\":" << readCmdWord(macroCmd.cmd, Aux0Word) <<
        ",\"aux1\":" << readCmdWord(macroCmd.cmd, Aux1Word);
}

void
SpecializedExecutionUnit::emitProfileBegin(MacroCmdContext &macroCmd) const
{
#ifdef NPU_PROFILE_ENABLE
    std::ostringstream os;
    macroCmd.profileBeginTick = curTick();
    appendProfileEventJson(os, "begin", macroCmd, macroCmd.profileBeginTick);
    const std::string payload = os.str();
    DPRINTF(NPUProfile, "NPU_PROFILE %s\n", payload.c_str());
#else
    (void)macroCmd;
#endif
}

void
SpecializedExecutionUnit::emitProfileEnd(const MacroCmdContext &macroCmd) const
{
#ifdef NPU_PROFILE_ENABLE
    std::ostringstream os;
    appendProfileEventJson(os, "end", macroCmd, curTick());
    const std::string payload = os.str();
    DPRINTF(NPUProfile, "NPU_PROFILE %s\n", payload.c_str());
#else
    (void)macroCmd;
#endif
}

void
SpecializedExecutionUnit::recordUopIssue(MacroCmdContext &macroCmd,
                                         const MicroOpContext &uop)
{
    UopProfileEvent event;
    event.kind = uop.kind;
    event.profileId = uop.profileId;
    event.token = uop.token;
    event.portId = uop.portId;
    event.addr = uop.addr;
    event.size = uop.size;
    event.issueTick = curTick();
    macroCmd.profileUops.push_back(event);
}

void
SpecializedExecutionUnit::recordUopComplete(MacroCmdContext &macroCmd,
                                            uint64_t profileId)
{
    for (auto it = macroCmd.profileUops.rbegin();
         it != macroCmd.profileUops.rend(); ++it) {
        if (it->profileId == profileId) {
            it->completeTick = curTick();
            return;
        }
    }
}

void
SpecializedExecutionUnit::appendProfileEventJson(
    std::ostream &os, const char *phase, const MacroCmdContext &macroCmd,
    Tick eventTick) const
{
    os << '{';
    os << "\"event\":";
    appendJsonString(os, phase);
    os << ",\"tick\":" << eventTick;
    os << ",\"seu_name\":";
    appendJsonString(os, name());
    os << ",\"seu_type\":";
    appendJsonString(os, profileSeuType());
    os << ",\"macro_id\":" << macroCmd.macroCmdId;
    os << ",\"issue_queue\":" << macroCmd.targetIssueQueueId;
    os << ",\"macro_kind\":";
    appendJsonString(os, macroCmdKindName(macroCmd.kind));
    os << ",\"device_type\":" <<
        static_cast<unsigned>(macroCmd.fields.deviceType);
    os << ",\"device_id\":" << static_cast<unsigned>(macroCmd.fields.deviceId);
    os << ",\"opcode\":" << static_cast<unsigned>(macroCmd.fields.opCode);
    os << ",\"sync_indicator\":" <<
        static_cast<unsigned>(macroCmd.fields.syncIndicator);
    os << ",\"set_indicator_sns\":" <<
        (macroCmd.fields.setIndicatorSns ? "true" : "false");
    os << ",\"set_indicator_snd\":" <<
        (macroCmd.fields.setIndicatorSnd ? "true" : "false");
    os << ",\"issued_load_uops\":" << macroCmd.issuedLoadUops;
    os << ",\"issued_store_uops\":" << macroCmd.issuedStoreUops;
    os << ",\"issued_exec_uops\":" << macroCmd.issuedExecUops;
    os << ",\"completed_load_uops\":" << macroCmd.completedLoadUops;
    os << ",\"completed_store_uops\":" << macroCmd.completedStoreUops;
    os << ",\"exec_active_ticks\":" << macroCmd.execActiveTicks;
    if (macroCmd.profileBeginTick != 0 &&
        eventTick >= macroCmd.profileBeginTick) {
        os << ",\"duration\":" << (eventTick - macroCmd.profileBeginTick);
    }
    os << ",\"raw_words\":";
    appendCmdWordsJson(os, macroCmd.cmd);
    os << ",\"details\":{";
    appendProfileDetailsJson(macroCmd, os);
    os << "}";
    if (std::string(phase) == "end") {
        os << ",\"uop_events\":";
        appendUopProfileEventsJson(os, macroCmd);
    }
    os << "}";
}

void
SpecializedExecutionUnit::appendUopProfileEventsJson(
    std::ostream &os, const MacroCmdContext &macroCmd) const
{
    os << '[';
    for (size_t i = 0; i < macroCmd.profileUops.size(); ++i) {
        const auto &uop = macroCmd.profileUops[i];
        if (i != 0) {
            os << ',';
        }
        os << '{';
        os << "\"profile_id\":" << uop.profileId;
        os << ",\"kind\":";
        appendJsonString(os, uopKindName(uop.kind));
        os << ",\"token\":" << uop.token;
        os << ",\"port\":" << uop.portId;
        os << ",\"addr\":" << static_cast<unsigned long long>(uop.addr);
        os << ",\"size\":" << uop.size;
        os << ",\"start_tick\":" << uop.issueTick;
        os << ",\"end_tick\":" << uop.completeTick;
        if (uop.completeTick >= uop.issueTick) {
            os << ",\"duration\":" << (uop.completeTick - uop.issueTick);
        } else {
            os << ",\"duration\":0";
        }
        os << '}';
    }
    os << ']';
}

void
SpecializedExecutionUnit::appendJsonString(std::ostream &os,
                                           const std::string &value) const
{
    os << '"';
    for (const char ch : value) {
        switch (ch) {
          case '\\':
            os << "\\\\";
            break;
          case '"':
            os << "\\\"";
            break;
          case '\n':
            os << "\\n";
            break;
          case '\r':
            os << "\\r";
            break;
          case '\t':
            os << "\\t";
            break;
          default:
            os << ch;
            break;
        }
    }
    os << '"';
}

void
SpecializedExecutionUnit::appendCmdWordsJson(
    std::ostream &os, const std::vector<uint8_t> &cmd) const
{
    os << '[';
    const size_t word_count = cmd.size() / sizeof(uint32_t);
    for (size_t i = 0; i < word_count; ++i) {
        if (i != 0) {
            os << ',';
        }
        os << readCmdWord(cmd, i);
    }
    os << ']';
}

const char *
SpecializedExecutionUnit::macroCmdKindName(MacroCmdKind kind) const
{
    switch (kind) {
      case MacroCmdKind::Load:
        return "Load";
      case MacroCmdKind::Exec:
        return "Exec";
      case MacroCmdKind::Store:
        return "Store";
    }

    panic("%s: unreachable macro command kind", name());
}

const char *
SpecializedExecutionUnit::uopKindName(MicroOpContext::Kind kind) const
{
    switch (kind) {
      case MicroOpContext::Kind::Load:
        return "load";
      case MicroOpContext::Kind::Exec:
        return "exec";
      case MicroOpContext::Kind::Store:
        return "store";
      case MicroOpContext::Kind::SyncWrite:
        return "sync_write";
    }

    panic("%s: unreachable uop kind", name());
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
    pendingCompletionSyncWords.push_back(word);
    trySendCompletionSyncWords();
}

void
SpecializedExecutionUnit::trySendCompletionSyncWords()
{
    while (!pendingCompletionSyncWords.empty()) {
        PortID port_id = InvalidPortID;
        for (PortID i = 0; i < static_cast<PortID>(memSidePorts.size()); ++i) {
            if (!getMemSidePort(i).isBlocked()) {
                port_id = i;
                break;
            }
        }
        if (port_id == InvalidPortID) {
            return;
        }

        const uint32_t word = pendingCompletionSyncWords.front();
        pendingCompletionSyncWords.pop_front();

        DPRINTF(SpecializedExecutionUnit,
                "completion sync write word=%#x port=%d\n", word, port_id);

        const auto data = packWord(word);
        MemTxnContext txn;
        txn.kind = MemTxnContext::Kind::SyncWrite;
        txn.macroCmdId = 0;
        txn.ownerIssueQueueId = 0;
        txn.portId = port_id;
        txn.token = nextMicroOpToken++;
        txn.addr = SyncIndicatorBase;
        txn.size = sizeof(uint32_t);
        sendTrackedPacket(txn, &data);
    }
}

void
SpecializedExecutionUnit::appendLoadUop(MacroCmdContext &macroCmd,
                                        Addr addr, size_t size)
{
    MicroOpContext uop;
    uop.kind = MicroOpContext::Kind::Load;
    uop.macroCmdId = macroCmd.macroCmdId;
    uop.ownerIssueQueueId = macroCmd.targetIssueQueueId;
    uop.portId = macroCmd.boundMemPortId.has_value() ?
        *macroCmd.boundMemPortId : InvalidPortID;
    uop.token = nextMicroOpToken++;
    uop.addr = addr;
    uop.size = size;
    macroCmd.uopQueue.push_back(std::move(uop));
}

void
SpecializedExecutionUnit::appendStoreUop(MacroCmdContext &macroCmd,
                                         Addr addr, size_t size,
                                         const std::vector<uint8_t> &data)
{
    MicroOpContext uop;
    uop.kind = MicroOpContext::Kind::Store;
    uop.macroCmdId = macroCmd.macroCmdId;
    uop.ownerIssueQueueId = macroCmd.targetIssueQueueId;
    uop.portId = macroCmd.boundMemPortId.has_value() ?
        *macroCmd.boundMemPortId : InvalidPortID;
    uop.token = nextMicroOpToken++;
    uop.addr = addr;
    uop.size = size;
    uop.data = data;
    macroCmd.uopQueue.push_back(std::move(uop));
}

void
SpecializedExecutionUnit::appendExecUop(MacroCmdContext &macroCmd,
                                        Tick latency)
{
    MicroOpContext uop;
    uop.kind = MicroOpContext::Kind::Exec;
    uop.macroCmdId = macroCmd.macroCmdId;
    uop.ownerIssueQueueId = macroCmd.targetIssueQueueId;
    uop.portId = InvalidPortID;
    uop.token = nextMicroOpToken++;
    uop.latency = latency;
    macroCmd.uopQueue.push_back(std::move(uop));
}

void
SpecializedExecutionUnit::markEpiloguePending(MacroCmdContext &macroCmd)
{
    if (macroCmd.waitingCallback || macroCmd.outstandingMemUops != 0) {
        return;
    }

    if (macroCmd.phase == Phase::Done ||
        macroCmd.phase == Phase::Epilogue ||
        macroCmd.phase == Phase::EpiloguePending) {
        if (macroCmd.phase == Phase::EpiloguePending &&
            !macroCmd.epilogueQueued) {
            macroCmd.epilogueQueued = true;
            pendingEpilogueCmdIds.push_back(macroCmd.macroCmdId);
        }
        return;
    }

    macroCmd.phase = Phase::EpiloguePending;
    if (!macroCmd.epilogueQueued) {
        macroCmd.epilogueQueued = true;
        pendingEpilogueCmdIds.push_back(macroCmd.macroCmdId);
    }
    tryScheduleIssue();
}

SpecializedExecutionUnit::IssueQueueState &
SpecializedExecutionUnit::getIssueQueue(uint32_t issueQueueId)
{
    for (auto &queue : issueQueues) {
        if (queue.issueQueueId == issueQueueId) {
            return queue;
        }
    }
    panic("%s: issue queue %u not found", name(), issueQueueId);
}

const SpecializedExecutionUnit::IssueQueueState &
SpecializedExecutionUnit::getIssueQueue(uint32_t issueQueueId) const
{
    for (const auto &queue : issueQueues) {
        if (queue.issueQueueId == issueQueueId) {
            return queue;
        }
    }
    panic("%s: issue queue %u not found", name(), issueQueueId);
}

bool
SpecializedExecutionUnit::issueQueueExists(uint32_t issueQueueId) const
{
    for (const auto &queue : issueQueues) {
        if (queue.issueQueueId == issueQueueId) {
            return true;
        }
    }
    return false;
}

PortID
SpecializedExecutionUnit::mappedMemPort(const MacroCmdContext &macroCmd) const
{
    panic_if(!macroCmd.boundMemPortId.has_value(),
             "%s: macro command %llu has no mapped mem port",
             name(), static_cast<unsigned long long>(macroCmd.macroCmdId));
    return *macroCmd.boundMemPortId;
}

bool
SpecializedExecutionUnit::canActivateExclusively(
    const MacroCmdContext &macroCmd) const
{
    if (!activeMemTxns.empty() || !activeExecUops.empty() ||
        !pendingEpilogueCmdIds.empty()) {
        return false;
    }

    for (const auto &queue : issueQueues) {
        if (queue.issueQueueId == macroCmd.targetIssueQueueId) {
            if (queue.activeMacroCmdId.has_value()) {
                return false;
            }
            if (!queue.waitingMacroCmdIds.empty() &&
                queue.waitingMacroCmdIds.front() != macroCmd.macroCmdId) {
                return false;
            }
            continue;
        }

        if (queue.activeMacroCmdId.has_value() ||
            !queue.waitingMacroCmdIds.empty()) {
            return false;
        }
    }

    return true;
}

uint32_t
SpecializedExecutionUnit::extractCmdWord(const std::vector<uint8_t> &cmd) const
{
    return readCmdWord(cmd, 0);
}

uint32_t
SpecializedExecutionUnit::readCmdWord(const std::vector<uint8_t> &cmd,
                                      size_t word_idx) const
{
    const size_t offset = word_idx * sizeof(uint32_t);
    if (cmd.size() < offset + sizeof(uint32_t)) {
        return 0;
    }

    uint32_t word = 0;
    std::memcpy(&word, cmd.data() + offset, sizeof(word));
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

uint64_t
SpecializedExecutionUnit::queueOccupancy() const
{
    uint64_t total = dispatchQueue.size();
    for (const auto &queue : issueQueues) {
        total += queue.waitingMacroCmdIds.size();
        total += queue.activeMacroCmdId.has_value() ? 1U : 0U;
    }
    return total;
}

uint64_t
SpecializedExecutionUnit::completedCmdCount() const
{
    return completedCount;
}

bool
SpecializedExecutionUnit::isIssueBusy() const
{
    return queueOccupancy() != 0 || !activeMemTxns.empty() ||
           !activeExecUops.empty() || !pendingEpilogueCmdIds.empty();
}

uint64_t
SpecializedExecutionUnit::completedReadRespCount() const
{
    return completedReadCount;
}

uint64_t
SpecializedExecutionUnit::completedWriteRespCount() const
{
    return completedWriteCount;
}

uint64_t
SpecializedExecutionUnit::completedIterationCount() const
{
    return completedIterationValue;
}

uint64_t
SpecializedExecutionUnit::prologueCount() const
{
    return prologueCountValue;
}

uint64_t
SpecializedExecutionUnit::executeCount() const
{
    return executeCountValue;
}

uint64_t
SpecializedExecutionUnit::epilogueCount() const
{
    return epilogueCountValue;
}

uint64_t
SpecializedExecutionUnit::maxActiveMicroOps() const
{
    return maxConcurrentMicroOpsValue;
}

uint64_t
SpecializedExecutionUnit::maxMemPortOutstanding() const
{
    uint64_t max_value = 0;
    for (const auto value : maxMemPortOutstandingValues) {
        max_value = std::max(max_value, value);
    }
    return max_value;
}

} // namespace gem5
