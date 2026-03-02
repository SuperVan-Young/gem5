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

#include "base/cprintf.hh"
#include "base/logging.hh"
#include "debug/MegaCmdQueue.hh"
#include "mem/packet.hh"
#include "sim/system.hh"

namespace gem5
{

MegaCmdQueue::CPUSidePort::CPUSidePort(
    const std::string &name, int id, MegaCmdQueue *owner)
    : ResponsePort(name), owner(owner), id(id), needRetry(false),
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

    if (!owner->handleRequest(pkt, id)) {
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

    trySendRetry();
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
    return owner->getAddrRanges();
}

void
MegaCmdQueue::CPUSidePort::trySendRetry()
{
    if (!needRetry || blockedRespPacket != nullptr) {
        return;
    }

    if (!owner->canPushMegaCmd()) {
        return;
    }

    needRetry = false;
    if (isConnected()) {
        sendRetryReq();
    }
}

MegaCmdQueue::MegaCmdQueue(const MegaCmdQueueParams &params)
    : ClockedObject(params),
      numInputPort(params.num_input_port),
      megaCmdWidth(params.mega_cmd_width),
      cmdQueueDepth(params.cmd_queue_depth),
      megaCmdBytes(megaCmdWidth / 8),
      baseAddr(params.base_addr),
      hasEnqueuedCmd(false),
      clearEnqueueGateEvent(
          [this]() { clearEnqueueGate(); },
          name() + ".clearEnqueueGateEvent")
{
    fatal_if(megaCmdWidth == 0 || megaCmdWidth % 8 != 0,
             "%s: mega_cmd_width must be non-zero and byte aligned", name());
    fatal_if(cmdQueueDepth == 0,
             "%s: cmd_queue_depth must be greater than zero", name());

    stagingBuffers.resize(numInputPort);

    for (int i = 0; i < numInputPort; ++i) {
        stagingBuffers[i].bytes.resize(megaCmdBytes, 0);
        cpuSidePorts.emplace_back(csprintf("%s.cpu_side[%d]", name(), i),
                                  i, this);
    }
}

void
MegaCmdQueue::init()
{
    ClockedObject::init();
    fatal_if(numInputPort != 1,
             "%s: currently only num_input_port == 1 is supported", name());

    for (auto &port : cpuSidePorts) {
        if (port.isConnected()) {
            port.sendRangeChange();
        }
    }
}

Port &
MegaCmdQueue::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "cpu_side" && idx < cpuSidePorts.size()) {
        return cpuSidePorts[idx];
    }

    return ClockedObject::getPort(if_name, idx);
}

AddrRangeList
MegaCmdQueue::getAddrRanges() const
{
    return {AddrRange(baseAddr, baseAddr + (2 * megaCmdBytes))};
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

    if (pkt->getAddr() < baseAddr) {
        DPRINTF(MegaCmdQueue,
                "reject addr=%#llx below base=%#llx\n",
                pkt->getAddr(), baseAddr);
        return false;
    }

    const Addr offset = pkt->getAddr() - baseAddr;
    if (!validMmioOffset(offset, pkt->getSize())) {
        DPRINTF(MegaCmdQueue,
                "reject invalid mmio range addr=%#llx size=%u\n",
                pkt->getAddr(), pkt->getSize());
        return false;
    }

    if (offset < megaCmdBytes) {
        const bool ok = writeDataChunk(port_id, offset, pkt);
        DPRINTF(MegaCmdQueue,
                "data write addr=%#llx off=%#llx size=%u accepted=%d\n",
                pkt->getAddr(), offset, pkt->getSize(), ok);
        return ok;
    }

    const uint64_t ctrl = pkt->getUintX(ByteOrder::little);
    DPRINTF(MegaCmdQueue,
            "control write addr=%#llx off=%#llx val=%llu\n",
            pkt->getAddr(), offset,
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

void
MegaCmdQueue::clearEnqueueGate()
{
    hasEnqueuedCmd = false;
    DPRINTF(MegaCmdQueue, "clear same-cycle push gate\n");
    trySendRetries();
}

void
MegaCmdQueue::trySendRetries()
{
    for (auto &port : cpuSidePorts) {
        port.trySendRetry();
    }
}

uint64_t
MegaCmdQueue::queueOccupancy() const
{
    return queue.size();
}

} // namespace gem5
