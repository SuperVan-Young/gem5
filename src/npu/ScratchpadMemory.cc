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

#include "npu/ScratchpadMemory.hh"

#include <algorithm>

#include "base/trace.hh"
#include "debug/Drain.hh"
#include "debug/ScratchpadMemory.hh"

namespace gem5
{

namespace npu
{

ScratchpadMemory::ScratchpadMemory(const ScratchpadMemoryParams &p) :
    AbstractMemory(p),
    port(name() + ".port", *this),
    latency(p.latency),
    bandwidth(p.bandwidth),
    pipelineDepth(p.pipeline_depth),
    pipelinePorts(p.pipeline_ports),
    pipelinePortStride(p.pipeline_port_stride),
    inflightRequests(0),
    nextServiceTick(p.pipeline_ports, 0),
    nextSeq(0),
    acceptedRequestsValue(0),
    pipelineFullRetriesValue(0),
    maxInflightRequestsValue(0),
    retryReq(false),
    retryResp(false),
    accessEvent([this]{ processAccess(); }, name() + ".accessEvent"),
    dequeueEvent([this]{ dequeue(); }, name())
{
    DPRINTF(ScratchpadMemory, "Created ScratchpadMemory with latency=%llu, "
            "bandwidth=%f ticks/byte pipeline_depth=%u\n", latency,
            bandwidth, pipelineDepth);
    panic_if(pipelineDepth == 0, "%s: pipeline_depth must be non-zero",
             name());
    panic_if(pipelinePorts == 0, "%s: pipeline_ports must be non-zero",
             name());
    panic_if(pipelinePortStride == 0,
             "%s: pipeline_port_stride must be non-zero", name());
}

void
ScratchpadMemory::init()
{
    AbstractMemory::init();

    // Allow unconnected memories for flexibility
    if (port.isConnected()) {
        port.sendRangeChange();
    }
}

Tick
ScratchpadMemory::recvAtomic(PacketPtr pkt)
{
    panic_if(pkt->cacheResponding(),
             "Should not see packets where cache is responding");

    // Perform the actual memory access
    access(pkt);

    DPRINTF(ScratchpadMemory, "Atomic access: addr=%#llx, size=%d, "
            "latency=%llu\n", pkt->getAddr(), pkt->getSize(), latency);

    return latency;
}

Tick
ScratchpadMemory::recvAtomicBackdoor(PacketPtr pkt, MemBackdoorPtr &_backdoor)
{
    Tick lat = recvAtomic(pkt);
    getBackdoor(_backdoor);
    return lat;
}

void
ScratchpadMemory::recvFunctional(PacketPtr pkt)
{
    pkt->pushLabel(name());

    // Perform functional access to the backing store
    functionalAccess(pkt);

    // Also check packets in our timing queues.
    bool done = false;
    auto a = accessQueue.begin();
    while (!done && a != accessQueue.end()) {
        done = pkt->trySatisfyFunctional(a->pkt);
        ++a;
    }

    auto p = packetQueue.begin();
    while (!done && p != packetQueue.end()) {
        done = pkt->trySatisfyFunctional(p->pkt);
        ++p;
    }

    pkt->popLabel();
}

void
ScratchpadMemory::recvMemBackdoorReq(const MemBackdoorReq &req,
        MemBackdoorPtr &_backdoor)
{
    getBackdoor(_backdoor);
}

bool
ScratchpadMemory::recvTimingReq(PacketPtr pkt)
{
    panic_if(pkt->cacheResponding(),
             "Should not see packets where cache is responding");

    panic_if(!(pkt->isRead() || pkt->isWrite()),
             "Should only see read and writes at SPM, saw %s to %#llx\n",
             pkt->cmdString(), pkt->getAddr());

    // Ignore requests if we have committed to retry
    if (retryReq) {
        DPRINTF(ScratchpadMemory, "Ignoring request while retry pending: "
                "addr=%#llx\n", pkt->getAddr());
        return false;
    }

    if (inflightRequests >= pipelineDepth) {
        DPRINTF(ScratchpadMemory, "SPM pipeline full, scheduling retry: "
                "addr=%#llx inflight=%u depth=%u\n", pkt->getAddr(),
                inflightRequests, pipelineDepth);
        retryReq = true;
        ++pipelineFullRetriesValue;
        return false;
    }

    // Calculate receive delay from header and payload delays
    Tick receive_delay = pkt->headerDelay + pkt->payloadDelay;
    pkt->headerDelay = pkt->payloadDelay = 0;

    Tick service_tick =
        schedulePipelineService(pkt, curTick() + receive_delay);

    const bool needs_response = pkt->needsResponse();
    Tick response_tick = service_tick + latency;
    const uint64_t seq = nextSeq++;

    ++inflightRequests;
    ++acceptedRequestsValue;
    maxInflightRequestsValue =
        std::max<uint64_t>(maxInflightRequestsValue, inflightRequests);

    auto access_it = accessQueue.begin();
    while (access_it != accessQueue.end() &&
           (access_it->tick < service_tick ||
            (access_it->tick == service_tick && access_it->seq < seq))) {
        ++access_it;
    }
    accessQueue.emplace(access_it, pkt, service_tick, response_tick, seq,
                        needs_response);

    DPRINTF(ScratchpadMemory, "Timing request accepted: addr=%#llx "
            "size=%d service=%llu response=%llu inflight=%u depth=%u "
            "needsResponse=%d\n", pkt->getAddr(),
            pkt->getSize(), service_tick, response_tick, inflightRequests,
            pipelineDepth, needs_response);

    scheduleAccess();
    return true;
}

unsigned
ScratchpadMemory::pipelinePort(Addr addr) const
{
    const Addr base = getAddrRange().start();
    if (addr < base) {
        return 0;
    }
    return ((addr - base) / pipelinePortStride) % pipelinePorts;
}

Tick
ScratchpadMemory::schedulePipelineService(PacketPtr pkt, Tick ready_tick)
{
    Tick service_tick = ready_tick;
    Addr cursor = pkt->getAddr();
    size_t remaining = pkt->getSize();

    while (remaining != 0) {
        const Addr base = getAddrRange().start();
        const Addr offset = cursor < base ? 0 : cursor - base;
        const uint32_t bank_offset = offset % pipelinePortStride;
        const size_t chunk = std::min<size_t>(
            remaining, pipelinePortStride - bank_offset);
        const unsigned pipe_port = pipelinePort(cursor);
        const Tick bank_ready =
            std::max(ready_tick, nextServiceTick[pipe_port]);
        const Tick duration = chunk * bandwidth;

        nextServiceTick[pipe_port] = bank_ready + duration;
        service_tick = std::max(service_tick, bank_ready);
        cursor += chunk;
        remaining -= chunk;
    }

    return service_tick;
}

void
ScratchpadMemory::scheduleAccess()
{
    if (accessQueue.empty()) {
        return;
    }

    const Tick when = std::max(accessQueue.front().tick, curTick());
    if (accessEvent.scheduled()) {
        reschedule(accessEvent, when, true);
    } else {
        schedule(accessEvent, when);
    }
}

void
ScratchpadMemory::scheduleDequeue()
{
    if (retryResp || packetQueue.empty()) {
        return;
    }

    const Tick when = std::max(packetQueue.front().tick, curTick());
    if (dequeueEvent.scheduled()) {
        reschedule(dequeueEvent, when, true);
    } else {
        schedule(dequeueEvent, when);
    }
}

void
ScratchpadMemory::processAccess()
{
    while (!accessQueue.empty() && accessQueue.front().tick <= curTick()) {
        const DeferredAccess deferred_access = accessQueue.front();
        accessQueue.pop_front();

        PacketPtr pkt = deferred_access.pkt;
        DPRINTF(ScratchpadMemory, "Processing access: addr=%#llx "
                "size=%d tick=%llu response=%llu seq=%llu\n",
                pkt->getAddr(), pkt->getSize(), deferred_access.tick,
                deferred_access.responseTick,
                static_cast<unsigned long long>(deferred_access.seq));

        access(pkt);

        if (deferred_access.needsResponse) {
            assert(pkt->isResponse());

            auto response_it = packetQueue.begin();
            while (response_it != packetQueue.end() &&
                   (response_it->tick < deferred_access.responseTick ||
                    (response_it->tick == deferred_access.responseTick &&
                     response_it->seq < deferred_access.seq))) {
                ++response_it;
            }
            packetQueue.emplace(response_it, pkt, deferred_access.responseTick,
                                deferred_access.seq);
            scheduleDequeue();
        } else {
            pendingDelete.reset(pkt);
            completeRequest();
        }
    }

    scheduleAccess();
}

void
ScratchpadMemory::trySendRetryReq()
{
    if (retryReq && inflightRequests < pipelineDepth) {
        retryReq = false;
        port.sendRetryReq();
    }
}

void
ScratchpadMemory::completeRequest()
{
    assert(inflightRequests > 0);
    --inflightRequests;
    trySendRetryReq();

    if (drainState() == DrainState::Draining && accessQueue.empty() &&
        packetQueue.empty() && inflightRequests == 0 && !retryResp) {
        DPRINTF(Drain, "Draining of ScratchpadMemory complete\n");
        signalDrainDone();
    }
}

void
ScratchpadMemory::dequeue()
{
    assert(!packetQueue.empty());

    DeferredPacket deferred_pkt = packetQueue.front();

    DPRINTF(ScratchpadMemory, "Dequeueing response: addr=%#llx, tick=%llu\n",
            deferred_pkt.pkt->getAddr(), deferred_pkt.tick);

    // Try to send the response
    retryResp = !port.sendTimingResp(deferred_pkt.pkt);

    if (!retryResp) {
        // Successfully sent, remove from queue
        packetQueue.pop_front();
        completeRequest();

        if (!packetQueue.empty()) {
            scheduleDequeue();
        }
    }
}

void
ScratchpadMemory::recvRespRetry()
{
    assert(retryResp);
    dequeue();
}

Port &
ScratchpadMemory::getPort(const std::string &if_name, PortID idx)
{
    if (if_name != "port") {
        return AbstractMemory::getPort(if_name, idx);
    }
    return port;
}

DrainState
ScratchpadMemory::drain()
{
    const bool has_pending_work =
        !accessQueue.empty() || !packetQueue.empty() ||
        inflightRequests != 0 || retryResp;
    if (has_pending_work) {
        DPRINTF(Drain, "ScratchpadMemory queue has requests, waiting\n");
        return DrainState::Draining;
    }
    return DrainState::Drained;
}

// MemoryPort implementation

ScratchpadMemory::MemoryPort::MemoryPort(const std::string& _name,
                                         ScratchpadMemory& _spm)
    : ResponsePort(_name), spm(_spm)
{}

AddrRangeList
ScratchpadMemory::MemoryPort::getAddrRanges() const
{
    AddrRangeList ranges;
    ranges.push_back(spm.getAddrRange());
    return ranges;
}

Tick
ScratchpadMemory::MemoryPort::recvAtomic(PacketPtr pkt)
{
    return spm.recvAtomic(pkt);
}

Tick
ScratchpadMemory::MemoryPort::recvAtomicBackdoor(
        PacketPtr pkt, MemBackdoorPtr &_backdoor)
{
    return spm.recvAtomicBackdoor(pkt, _backdoor);
}

void
ScratchpadMemory::MemoryPort::recvFunctional(PacketPtr pkt)
{
    spm.recvFunctional(pkt);
}

void
ScratchpadMemory::MemoryPort::recvMemBackdoorReq(const MemBackdoorReq &req,
        MemBackdoorPtr &backdoor)
{
    spm.recvMemBackdoorReq(req, backdoor);
}

bool
ScratchpadMemory::MemoryPort::recvTimingReq(PacketPtr pkt)
{
    return spm.recvTimingReq(pkt);
}

void
ScratchpadMemory::MemoryPort::recvRespRetry()
{
    spm.recvRespRetry();
}

} // namespace npu
} // namespace gem5
