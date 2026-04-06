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
    isBusy(false),
    retryReq(false),
    retryResp(false),
    releaseEvent([this]{ release(); }, name()),
    dequeueEvent([this]{ dequeue(); }, name())
{
    DPRINTF(ScratchpadMemory, "Created ScratchpadMemory with latency=%llu, "
            "bandwidth=%f ticks/byte\n", latency, bandwidth);
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

    // Also check packets in our timing queue
    bool done = false;
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

    // If busy, remember to retry
    if (isBusy) {
        DPRINTF(ScratchpadMemory, "SPM busy, scheduling retry: addr=%#llx\n",
                pkt->getAddr());
        retryReq = true;
        return false;
    }

    // Calculate receive delay from header and payload delays
    Tick receive_delay = pkt->headerDelay + pkt->payloadDelay;
    pkt->headerDelay = pkt->payloadDelay = 0;

    // Calculate bandwidth-limited duration
    Tick duration = pkt->getSize() * bandwidth;

    // Only schedule release event if there's actual bandwidth constraint
    if (duration != 0) {
        schedule(releaseEvent, curTick() + duration);
        isBusy = true;
    }

    // Process the packet (perform access)
    bool needsResponse = pkt->needsResponse();
    recvAtomic(pkt);

    DPRINTF(ScratchpadMemory, "Timing request processed: addr=%#llx, "
            "size=%d, needsResponse=%d\n", pkt->getAddr(), pkt->getSize(),
            needsResponse);

    // Turn packet around if response expected
    if (needsResponse) {
        assert(pkt->isResponse());

        // Calculate when to send the response
        Tick when_to_send = curTick() + receive_delay + latency;

        // Insert in sorted order, but maintain order with same-address packets
        auto i = packetQueue.end();
        if (!packetQueue.empty()) {
            --i;
            while (i != packetQueue.begin() && when_to_send < i->tick &&
                   !i->pkt->matchAddr(pkt)) {
                --i;
            }
            ++i;
        }

        packetQueue.emplace(i, pkt, when_to_send);

        // Schedule dequeue event if not already pending
        if (!retryResp && !dequeueEvent.scheduled()) {
            schedule(dequeueEvent, packetQueue.back().tick);
        }
    } else {
        // No response needed, delete packet
        pendingDelete.reset(pkt);
    }

    return true;
}

void
ScratchpadMemory::release()
{
    assert(isBusy);
    isBusy = false;

    DPRINTF(ScratchpadMemory, "SPM released from busy state\n");

    if (retryReq) {
        retryReq = false;
        port.sendRetryReq();
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

        if (!packetQueue.empty()) {
            // Schedule next dequeue
            reschedule(dequeueEvent,
                       std::max(packetQueue.front().tick, curTick()), true);
        } else if (drainState() == DrainState::Draining) {
            DPRINTF(Drain, "Draining of ScratchpadMemory complete\n");
            signalDrainDone();
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
    if (!packetQueue.empty()) {
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
