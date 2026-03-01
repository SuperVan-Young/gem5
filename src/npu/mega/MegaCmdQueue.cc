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
#include "mem/packet.hh"
#include "sim/system.hh"

namespace gem5
{

MegaCmdQueue::CPUSidePort::CPUSidePort(
    const std::string &name, int id, MegaCmdQueue *owner)
    : ResponsePort(name), owner(owner), id(id), needRetry(false),
      blockedRespPacket(nullptr)
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
        if (!sendTimingResp(pkt)) {
            blockedRespPacket = pkt;
        }
    } else {
        delete pkt;
    }

    return true;
}

void
MegaCmdQueue::CPUSidePort::recvRespRetry()
{
    assert(blockedRespPacket != nullptr);

    PacketPtr pkt = blockedRespPacket;
    blockedRespPacket = nullptr;

    if (!sendTimingResp(pkt)) {
        blockedRespPacket = pkt;
        return;
    }

    trySendRetry();
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

    if (!owner->canAcceptDoorbell()) {
        return;
    }

    needRetry = false;
    if (isConnected()) {
        sendRetryReq();
    }
}

MegaCmdQueue::MegaCmdQueue(const MegaCmdQueueParams &params)
    : SimObject(params),
      numInputPort(params.num_input_port),
      megaCmdWidth(params.mega_cmd_width),
      cmdQueueDepth(params.cmd_queue_depth),
      megaCmdBytes(megaCmdWidth / 8)
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
    SimObject::init();
    fatal_if(numInputPort != 1,
             "%s: currently only num_input_port == 1 is supported", name());
}

Port &
MegaCmdQueue::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "cpu_side" && idx < cpuSidePorts.size()) {
        return cpuSidePorts[idx];
    }

    return SimObject::getPort(if_name, idx);
}

AddrRangeList
MegaCmdQueue::getAddrRanges() const
{
    return {AddrRange(0, MaxAddrSpace)};
}

bool
MegaCmdQueue::canAcceptDoorbell() const
{
    return queue.size() < cmdQueueDepth;
}

bool
MegaCmdQueue::appendDataBytes(PortID port_id, const uint8_t *src, size_t size)
{
    auto &staging = stagingBuffers[port_id];
    if (staging.writeOffset + size > megaCmdBytes) {
        return false;
    }

    std::copy(src, src + size, staging.bytes.begin() + staging.writeOffset);
    staging.writeOffset += size;
    return true;
}

bool
MegaCmdQueue::appendDataChunk(PortID port_id, PacketPtr pkt)
{
    return appendDataBytes(port_id, pkt->getConstPtr<uint8_t>(), pkt->getSize());
}

bool
MegaCmdQueue::enqueueStagedCommand(PortID port_id)
{
    auto &staging = stagingBuffers[port_id];
    if (staging.writeOffset != megaCmdBytes) {
        return false;
    }

    if (queue.size() >= cmdQueueDepth) {
        return false;
    }

    queue.emplace_back(staging.bytes.begin(), staging.bytes.end());
    staging.writeOffset = 0;
    std::fill(staging.bytes.begin(), staging.bytes.end(), 0);
    return true;
}

bool
MegaCmdQueue::handleRequest(PacketPtr pkt, PortID port_id)
{
    if (!pkt->isWrite()) {
        return false;
    }

    if (pkt->getAddr() == DoorbellOffset) {
        return enqueueStagedCommand(port_id);
    }

    return appendDataChunk(port_id, pkt);
}

void
MegaCmdQueue::trySendRetries()
{
    for (auto &port : cpuSidePorts) {
        port.trySendRetry();
    }
}

bool
MegaCmdQueue::testWriteWord(uint64_t data_addr, uint32_t value)
{
    uint8_t data[sizeof(value)] = {};
    std::memcpy(data, &value, sizeof(value));

    if (data_addr == DoorbellOffset) {
        return false;
    }

    return appendDataBytes(0, data, sizeof(value));
}

bool
MegaCmdQueue::testRingDoorbell()
{
    return enqueueStagedCommand(0);
}

bool
MegaCmdQueue::popCmd()
{
    if (queue.empty()) {
        return false;
    }

    const bool was_full = queue.size() == cmdQueueDepth;
    queue.pop_front();

    if (was_full) {
        trySendRetries();
    }

    return true;
}

uint64_t
MegaCmdQueue::queueOccupancy() const
{
    return queue.size();
}

} // namespace gem5
