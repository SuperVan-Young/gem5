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

#include <cstring>

#include "base/trace.hh"
#include "debug/SpecializedExecutionUnit.hh"
#include "sim/system.hh"

namespace gem5
{

namespace
{

constexpr Addr SyncIndicatorBase = 0x71000000;

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
      memSidePort(params.name + ".mem_side", this),
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
    stagingBuffer.bytes.resize(macroCmdBytes, 0);
    DPRINTF(SpecializedExecutionUnit,
            "Created SEU: base_addr=%#x cmd_bytes=%u queue_depth=%u\n",
            baseAddr, macroCmdBytes, cmdQueueDepth);
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
}

Port &
SpecializedExecutionUnit::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "cpu_side") {
        return cpuSidePort;
    } else if (if_name == "mem_side") {
        return memSidePort;
    }
    return ClockedObject::getPort(if_name, idx);
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
    Tick execLatency = process(cmd);
    schedule(finishExecutionEvent, curTick() + execLatency);
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
SpecializedExecutionUnit::sendMemRequest(PacketPtr pkt)
{
    panic_if(
        activeMemPacket != nullptr,
        "%s: sendMemRequest requested while memory packet is still active",
        name());
    activeMemPacket = pkt;
    memSidePort.sendPacket(activeMemPacket);
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
    RequestPtr req = std::make_shared<Request>(
        SyncIndicatorBase, sizeof(uint32_t), Request::Flags(),
        Request::funcRequestorId);
    PacketPtr pkt = new Packet(req, MemCmd::WriteReq);
    pkt->allocate();
    pkt->setData(reinterpret_cast<const uint8_t *>(&word));

    DPRINTF(
        SpecializedExecutionUnit, "completion sync write word=%#x\n", word);
    sendMemRequest(pkt);
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
                                             uint8_t *buffer)
{
    // First version: not implemented
    return false;
}

bool
SpecializedExecutionUnit::startBlockingWrite(Addr addr, size_t size,
                                              const uint8_t *buffer)
{
    // First version: not implemented
    return false;
}

bool
SpecializedExecutionUnit::handleMemResponse(PacketPtr pkt)
{
    DPRINTF(SpecializedExecutionUnit,
            "Received memory response for addr=%#x\n", pkt->getAddr());

    cleanupActiveMemPacket();
    if (!issueCmdBusy && !cmdQueue.empty()) {
        tryScheduleIssue();
    }
    return true;
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

} // namespace gem5
