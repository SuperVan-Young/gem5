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

#include "npu/mega/MegaCmdQueue.hh"

#include <algorithm>
#include <cstring>
#include <memory>

#include "base/cprintf.hh"
#include "base/logging.hh"
#include "debug/MegaCmdQueue.hh"
#include "mem/packet.hh"
#include "sim/system.hh"

namespace gem5
{

namespace
{

constexpr Addr MmioBase = 0x70000000;
constexpr Addr SyncIndicatorBase = 0x71000000;

uint32_t
extractCmdWord(const std::vector<uint8_t> &cmd)
{
    if (cmd.size() < sizeof(uint32_t)) {
        return 0;
    }

    uint32_t word = 0;
    std::memcpy(&word, cmd.data(), sizeof(word));
    return word;
}

} // anonymous namespace

MegaCmdQueue::CPUSidePort::CPUSidePort(const std::string &name, PortID id,
                                       bool sync_indicator_port,
                                       MegaCmdQueue *owner)
    : ResponsePort(name), owner(owner), id(id),
      syncIndicatorPort(sync_indicator_port), needRetry(false),
      blockedRespPacket(nullptr),
      sendResponseEvent([this]() { sendDeferredResponse(); },
                        name + ".sendResponseEvent")
{
}

bool
MegaCmdQueue::CPUSidePort::recvTimingReq(PacketPtr pkt)
{
    if (blockedRespPacket != nullptr || needRetry) {
        needRetry = true;
        return false;
    }

    const bool accepted = syncIndicatorPort ?
        owner->handleSyncIndicatorRequest(pkt) : owner->handleRequest(pkt, id);
    if (!accepted) {
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
MegaCmdQueue::CPUSidePort::sendDeferredResponse()
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
}

void
MegaCmdQueue::CPUSidePort::recvRespRetry()
{
    assert(blockedRespPacket != nullptr);
    sendDeferredResponse();
}

AddrRangeList
MegaCmdQueue::CPUSidePort::getAddrRanges() const
{
    return syncIndicatorPort ? owner->getSyncIndicatorAddrRanges() :
                               owner->getCpuAddrRanges(id);
}

void
MegaCmdQueue::CPUSidePort::trySendRetry()
{
    if (!needRetry || blockedRespPacket != nullptr) {
        return;
    }

    if (!syncIndicatorPort && !owner->canPushMegaCmd()) {
        return;
    }

    needRetry = false;
    if (isConnected()) {
        sendRetryReq();
    }
}

MegaCmdQueue::MemSidePort::MemSidePort(
    const std::string &name, MegaCmdQueue *owner)
    : RequestPort(name, owner), owner(owner)
{
}

bool
MegaCmdQueue::MemSidePort::sendPacket(PacketPtr pkt)
{
    return sendTimingReq(pkt);
}

bool
MegaCmdQueue::MemSidePort::recvTimingResp(PacketPtr pkt)
{
    return owner->handleMemResponse(pkt);
}

void
MegaCmdQueue::MemSidePort::recvReqRetry()
{
    owner->retryDispatch();
}

MegaCmdQueue::MegaCmdQueue(const MegaCmdQueueParams &params)
    : ClockedObject(params),
      syncIndicatorSidePort(params.name + ".sync_indicator_side", InvalidPortID,
                            true, this),
      memSidePort(params.name + ".mem_side", this),
      numInputPort(params.num_input_port),
      megaCmdWidth(params.mega_cmd_width),
      cmdQueueDepth(params.cmd_queue_depth),
      megaCmdBytes(megaCmdWidth / 8),
      baseAddr(params.base_addr),
      rangeAddr(params.range_addr),
      numSyncIndicator(params.num_sync_indicator),
      syncIndicatorTable(numSyncIndicator, 0),
      hasEnqueuedCmd(false),
      writeInFlight(false),
      writeAwaitingRetry(false),
      writePacket(nullptr),
      clearEnqueueGateEvent(
          [this]() { clearEnqueueGate(); },
          name() + ".clearEnqueueGateEvent")
{
    fatal_if(megaCmdWidth == 0 || megaCmdWidth % 8 != 0,
             "%s: mega_cmd_width must be non-zero and byte aligned", name());
    fatal_if(cmdQueueDepth == 0,
             "%s: cmd_queue_depth must be greater than zero", name());
    fatal_if(numInputPort == 0,
             "%s: num_input_port must be greater than zero", name());
    fatal_if(rangeAddr <= baseAddr,
             "%s: range_addr must be above base_addr", name());

    stagingBuffers.resize(numInputPort);
    cpuSidePorts.reserve(numInputPort);

    for (PortID i = 0; i < numInputPort; ++i) {
        stagingBuffers[i].bytes.resize(megaCmdBytes, 0);
        cpuSidePorts.emplace_back(csprintf("%s.cpu_side[%d]", name(), i),
                                  i, false, this);

        const Addr start = portBaseAddr(i);
        const Addr end = start + (2 * megaCmdBytes);
        fatal_if(end > rangeAddr,
                 "%s: cpu_side[%d] MMIO range [%#llx, %#llx) exceeds "
                 "range_addr=%#llx",
                 name(), i, start, end, rangeAddr);
    }

    fatal_if(SyncIndicatorBase + sizeof(uint32_t) > rangeAddr,
             "%s: sync indicator MMIO [%#llx, %#llx) exceeds range_addr=%#llx",
             name(), SyncIndicatorBase,
             SyncIndicatorBase + sizeof(uint32_t), rangeAddr);
}

MegaCmdQueue::~MegaCmdQueue()
{
    cleanupWritePacket();
}

void
MegaCmdQueue::cleanupWritePacket()
{
    if (writePacket != nullptr) {
        delete writePacket;
        writePacket = nullptr;
    }
}

void
MegaCmdQueue::init()
{
    ClockedObject::init();

    for (auto &port : cpuSidePorts) {
        if (port.isConnected()) {
            port.sendRangeChange();
        }
    }

    if (syncIndicatorSidePort.isConnected()) {
        syncIndicatorSidePort.sendRangeChange();
    }
}

Port &
MegaCmdQueue::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "cpu_side" && idx < cpuSidePorts.size()) {
        return cpuSidePorts[idx];
    }

    if (if_name == "sync_indicator_side") {
        return syncIndicatorSidePort;
    }

    if (if_name == "mem_side") {
        return memSidePort;
    }

    return ClockedObject::getPort(if_name, idx);
}

Addr
MegaCmdQueue::portBaseAddr(PortID port_id) const
{
    return baseAddr + (static_cast<Addr>(port_id) << 20);
}

AddrRangeList
MegaCmdQueue::getCpuAddrRanges(PortID port_id) const
{
    const Addr port_base = portBaseAddr(port_id);
    return {AddrRange(port_base, port_base + (2 * megaCmdBytes))};
}

AddrRangeList
MegaCmdQueue::getSyncIndicatorAddrRanges() const
{
    return {AddrRange(SyncIndicatorBase, SyncIndicatorBase + sizeof(uint32_t))};
}

bool
MegaCmdQueue::canPushMegaCmd() const
{
    return queue.size() < cmdQueueDepth && !hasEnqueuedCmd;
}

bool
MegaCmdQueue::validMmioOffset(Addr offset, size_t size) const
{
    const Addr window_size = 2 * megaCmdBytes;
    if (offset >= window_size) {
        return false;
    }

    if (size == 0) {
        return false;
    }

    return offset + size <= window_size;
}

bool
MegaCmdQueue::writeDataBytes(PortID port_id, Addr offset,
                             const uint8_t *src, size_t size)
{
    auto &staging = stagingBuffers[port_id];
    if (offset + size > megaCmdBytes) {
        return false;
    }

    std::copy(src, src + size, staging.bytes.begin() + offset);
    return true;
}

bool
MegaCmdQueue::writeDataChunk(PortID port_id, Addr offset, PacketPtr pkt)
{
    return writeDataBytes(
        port_id, offset, pkt->getConstPtr<uint8_t>(), pkt->getSize());
}

bool
MegaCmdQueue::recvTimingPushReq(PortID port_id)
{
    if (!canPushMegaCmd()) {
        DPRINTF(MegaCmdQueue,
                "push rejected: queue=%llu depth=%u hasEnqueued=%d\n",
                static_cast<unsigned long long>(queue.size()),
                cmdQueueDepth, hasEnqueuedCmd);
        return false;
    }

    queue.emplace_back(stagingBuffers[port_id].bytes.begin(),
                       stagingBuffers[port_id].bytes.end());
    std::fill(stagingBuffers[port_id].bytes.begin(),
              stagingBuffers[port_id].bytes.end(), 0);

    hasEnqueuedCmd = true;
    if (!clearEnqueueGateEvent.scheduled()) {
        schedule(clearEnqueueGateEvent, clockEdge(Cycles(1)));
    }

    DPRINTF(MegaCmdQueue,
            "push accepted: queue=%llu/%u clear_tick=%llu\n",
            static_cast<unsigned long long>(queue.size()), cmdQueueDepth,
            static_cast<unsigned long long>(clockEdge(Cycles(1))));

    tryDispatchNext();
    return true;
}

void
MegaCmdQueue::popMegaCmd()
{
    panic_if(queue.empty(), "%s: pop requested on empty queue", name());

    const bool was_blocked = !canPushMegaCmd();
    queue.pop_front();

    DPRINTF(MegaCmdQueue,
            "pop executed: queue=%llu/%u hasEnqueued=%d\n",
            static_cast<unsigned long long>(queue.size()), cmdQueueDepth,
            hasEnqueuedCmd);

    if (was_blocked && canPushMegaCmd()) {
        trySendRetries();
    }

    tryDispatchNext();
}

bool
MegaCmdQueue::recvTimingPopReq()
{
    popMegaCmd();
    return true;
}

bool
MegaCmdQueue::handleRequest(PacketPtr pkt, PortID port_id)
{
    if (!pkt->isWrite()) {
        DPRINTF(MegaCmdQueue, "reject non-write req cmd=%s\n", pkt->cmdString());
        return false;
    }

    const Addr port_base = portBaseAddr(port_id);
    if (pkt->getAddr() < port_base || pkt->getAddr() >= rangeAddr) {
        DPRINTF(MegaCmdQueue,
                "reject addr=%#llx outside cpu_side[%d] range\n",
                pkt->getAddr(), port_id);
        return false;
    }

    const Addr offset = pkt->getAddr() - port_base;
    if (!validMmioOffset(offset, pkt->getSize())) {
        DPRINTF(MegaCmdQueue,
                "reject invalid mmio range addr=%#llx size=%u\n",
                pkt->getAddr(), pkt->getSize());
        return false;
    }

    if (offset < megaCmdBytes) {
        const bool ok = writeDataChunk(port_id, offset, pkt);
        DPRINTF(MegaCmdQueue,
                "data write port=%d addr=%#llx off=%#llx size=%u accepted=%d\n",
                port_id, pkt->getAddr(), offset, pkt->getSize(), ok);
        return ok;
    }

    const uint64_t ctrl = pkt->getUintX(ByteOrder::little);
    DPRINTF(MegaCmdQueue,
            "control write port=%d addr=%#llx off=%#llx val=%llu\n",
            port_id, pkt->getAddr(), offset,
            static_cast<unsigned long long>(ctrl));

    if (ctrl == 0) {
        return recvTimingPushReq(port_id);
    }

    if (ctrl == 1) {
        return recvTimingPopReq();
    }

    DPRINTF(MegaCmdQueue, "reject control val=%llu\n",
            static_cast<unsigned long long>(ctrl));
    return false;
}

bool
MegaCmdQueue::handleSyncIndicatorRequest(PacketPtr pkt)
{
    if (!pkt->isWrite()) {
        DPRINTF(MegaCmdQueue,
                "reject sync-indicator non-write req cmd=%s\n",
                pkt->cmdString());
        return false;
    }

    const Addr addr = pkt->getAddr();
    if (addr < SyncIndicatorBase || addr >= rangeAddr) {
        DPRINTF(MegaCmdQueue,
                "reject sync-indicator addr=%#llx outside range\n", addr);
        return false;
    }

    if (pkt->getSize() < sizeof(uint32_t)) {
        DPRINTF(MegaCmdQueue,
                "reject sync-indicator write size=%u (<4)\n", pkt->getSize());
        return false;
    }

    uint32_t word = 0;
    std::memcpy(&word, pkt->getConstPtr<uint8_t>(), sizeof(uint32_t));
    const CmdFields fields = parseCmdFields(word);

    if (fields.opCode != 1) {
        DPRINTF(MegaCmdQueue,
                "reject sync-indicator op=%u idx=%u (only op=1 supported)\n",
                fields.opCode, fields.indicatorIdx);
        return false;
    }

    if (fields.indicatorIdx >= numSyncIndicator) {
        DPRINTF(MegaCmdQueue,
                "reject sync-indicator idx=%u out of range [0, %u)\n",
                fields.indicatorIdx, numSyncIndicator);
        return false;
    }

    syncIndicatorTable[fields.indicatorIdx] = 1;
    DPRINTF(MegaCmdQueue,
            "sync-indicator set idx=%u (device_type=%u device_id=%u)\n",
            fields.indicatorIdx, fields.deviceType, fields.deviceId);

    tryDispatchNext();
    return true;
}

void
MegaCmdQueue::clearEnqueueGate()
{
    hasEnqueuedCmd = false;
    DPRINTF(MegaCmdQueue, "clear same-cycle push gate\n");
    tryDispatchNext();
    trySendRetries();
}

void
MegaCmdQueue::trySendRetries()
{
    for (auto &port : cpuSidePorts) {
        port.trySendRetry();
    }

    syncIndicatorSidePort.trySendRetry();
}

MegaCmdQueue::CmdFields
MegaCmdQueue::parseCmdFields(uint32_t word) const
{
    CmdFields fields;
    fields.deviceType = (word >> 24) & 0xF;
    fields.deviceId = (word >> 20) & 0xF;
    fields.opCode = (word >> 16) & 0xF;
    fields.indicatorIdx = word & 0xFFFF;
    return fields;
}

MegaCmdQueue::CmdFields
MegaCmdQueue::parseCmdFields(const std::vector<uint8_t> &cmd) const
{
    return parseCmdFields(extractCmdWord(cmd));
}

bool
MegaCmdQueue::tryDispatchNext()
{
    if (!memSidePort.isConnected() ||
        writeInFlight || writeAwaitingRetry || writePacket != nullptr ||
        queue.empty()) {
        return false;
    }

    const auto &cmd = queue.front();
    const CmdFields fields = parseCmdFields(cmd);
    if (fields.deviceType == 0x1 && fields.opCode == 0) {
        if (fields.indicatorIdx >= numSyncIndicator) {
            DPRINTF(MegaCmdQueue,
                    "sync wait blocked by invalid idx=%u (table size=%u)\n",
                    fields.indicatorIdx, numSyncIndicator);
            return false;
        }

        if (!syncIndicatorTable[fields.indicatorIdx]) {
            DPRINTF(MegaCmdQueue,
                    "sync wait blocked idx=%u indicator=0 queue=%llu\n",
                    fields.indicatorIdx,
                    static_cast<unsigned long long>(queue.size()));
            return false;
        }

        syncIndicatorTable[fields.indicatorIdx] = 0;
        DPRINTF(MegaCmdQueue,
                "sync wait released idx=%u indicator cleared\n",
                fields.indicatorIdx);
        popMegaCmd();
        return true;
    }

    const Addr target_addr = buildTargetAddr(cmd);

    RequestPtr req = std::make_shared<Request>(
        target_addr, megaCmdBytes, Request::Flags(), Request::funcRequestorId);
    writePacket = new Packet(req, MemCmd::WriteReq);
    writePacket->allocate();
    writePacket->setData(cmd.data());

    if (!memSidePort.sendPacket(writePacket)) {
        writeAwaitingRetry = true;
        DPRINTF(MegaCmdQueue,
                "dispatch blocked: target=%#llx queue=%llu\n",
                target_addr,
                static_cast<unsigned long long>(queue.size()));
        return false;
    }

    writeInFlight = true;

    DPRINTF(MegaCmdQueue,
            "dispatch sent: target=%#llx queue=%llu inflight=1\n",
            target_addr,
            static_cast<unsigned long long>(queue.size()));

    return true;
}

void
MegaCmdQueue::retryDispatch()
{
    if (!writeAwaitingRetry || writePacket == nullptr || writeInFlight) {
        return;
    }

    if (!memSidePort.sendPacket(writePacket)) {
        return;
    }

    writeAwaitingRetry = false;
    writeInFlight = true;

    DPRINTF(MegaCmdQueue,
            "dispatch sent on retry: target=%#llx queue=%llu inflight=1\n",
            writePacket->getAddr(),
            static_cast<unsigned long long>(queue.size()));
}

bool
MegaCmdQueue::handleMemResponse(PacketPtr pkt)
{
    panic_if(!writeInFlight || writePacket == nullptr,
             "%s: unexpected mem response without in-flight write", name());
    panic_if(pkt != writePacket,
             "%s: response packet mismatch", name());
    panic_if(queue.empty(),
             "%s: response arrived but queue is empty", name());

    const bool was_blocked = !canPushMegaCmd();

    DPRINTF(MegaCmdQueue,
            "dispatch complete: target=%#llx queue=%llu\n",
            pkt->getAddr(),
            static_cast<unsigned long long>(queue.size()));

    queue.pop_front();

    cleanupWritePacket();
    writeInFlight = false;
    writeAwaitingRetry = false;

    if (was_blocked && canPushMegaCmd()) {
        trySendRetries();
    }

    tryDispatchNext();
    return true;
}

Addr
MegaCmdQueue::buildTargetAddr(const std::vector<uint8_t> &cmd) const
{
    const CmdFields fields = parseCmdFields(cmd);
    return MmioBase | (static_cast<Addr>(fields.deviceType) << 24) |
           (static_cast<Addr>(fields.deviceId) << 20);
}

uint64_t
MegaCmdQueue::queueOccupancy() const
{
    return queue.size();
}

} // namespace gem5
