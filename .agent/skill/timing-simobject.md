# timing-simobject skill

## Purpose

Use this skill when asked to design, review, explain, or implement a gem5 SimObject that participates in **timing-mode** memory simulation.

This skill is specifically for modules built around gem5 memory ports and timing protocol semantics, such as objects similar to:

- `src/learning_gem5/part2/simple_memobj.*`
- `src/learning_gem5/part2/simple_cache.*`
- `src/mem/mem_delay.hh`
- `src/mem/cache/base.hh`
- `src/mem/xbar.hh`

## Core mental model

A timing-mode SimObject in gem5 is not just a read/write handler. It is a **transaction state machine** built on top of `RequestPort` / `ResponsePort`, backpressure, retries, packet lifetime rules, and event scheduling.

In timing mode:

- requests and responses do **not** complete immediately,
- `recvTiming*` may return `false` to indicate temporary backpressure,
- the sender must hold the packet and retry only after the corresponding retry callback,
- delays should be modeled with events and ticks, not immediate execution,
- packet ownership and blocked state must be carefully tracked.

Reference definitions:

- `src/sim/sim_object.hh:250`
- `src/sim/system.hh:270`
- `src/mem/port.hh:242`
- `src/mem/port.hh:338`

## What to explain first

When using this skill, begin by clarifying the role of the object:

1. Is it a **responder** to upstream CPU/cache requests?
2. Is it a **requester** toward downstream memory/interconnect?
3. Is it **blocking** (single outstanding transaction) or **non-blocking**?
4. Does it need only timing behavior, or also atomic / functional behavior?
5. Does it need snooping, coherence, or range propagation?

If the user asks for implementation help, inspect the existing code first and map the object to one of these common shapes:

- forwarding object,
- blocking cache-like object,
- bridge / delay object,
- memory controller endpoint,
- interconnect / routing object.

## Minimum method checklist

For a simple timing object with a CPU-side response port and a memory-side request port, the common minimum set is:

### CPU-side `ResponsePort`

Usually implements:

- `AddrRangeList getAddrRanges() const`
- `Tick recvAtomic(PacketPtr pkt)`
- `void recvFunctional(PacketPtr pkt)`
- `bool recvTimingReq(PacketPtr pkt)`
- `void recvRespRetry()`

Canonical examples:

- `src/learning_gem5/part2/simple_memobj.hh:54`
- `src/learning_gem5/part2/simple_cache.hh:57`

### Memory-side `RequestPort`

Usually implements:

- `bool recvTimingResp(PacketPtr pkt)`
- `void recvReqRetry()`
- `void recvRangeChange()`

Canonical examples:

- `src/learning_gem5/part2/simple_memobj.hh:136`
- `src/learning_gem5/part2/simple_cache.hh:143`

### Owner-side object logic

Usually needs methods like:

- `handleRequest(PacketPtr pkt, ...)`
- `handleResponse(PacketPtr pkt)`
- `handleFunctional(PacketPtr pkt)`
- `getAddrRanges() const`
- `sendRangeChange()`
- optional delayed handler such as `accessTiming(PacketPtr pkt)`

Canonical examples:

- `src/learning_gem5/part2/simple_memobj.hh:184`
- `src/learning_gem5/part2/simple_cache.hh:192`

## Port-role rules

Always explain port responsibilities explicitly.

### `ResponsePort`

A response port generally:

- **receives timing requests** via `recvTimingReq`,
- **sends timing responses** via `sendTimingResp`,
- receives `recvRespRetry` if a response send failed,
- provides `getAddrRanges()`.

### `RequestPort`

A request port generally:

- **sends timing requests** via `sendTimingReq`,
- **receives timing responses** via `recvTimingResp`,
- receives `recvReqRetry` if a request send failed,
- may receive `recvRangeChange()` from the downstream responder.

Reference semantics:

- `src/mem/port.hh:245`
- `src/mem/port.hh:443`
- `src/mem/port.hh:485`

## Required timing-mode implementation rules

When helping write or review a timing SimObject, check all of the following.

### 1. Backpressure must be handled correctly

If `sendTimingReq(pkt)` or `sendTimingResp(pkt)` returns `false`, the packet was **not accepted**.

Required behavior:

- keep ownership of the packet,
- store it in a blocked slot or queue,
- retry only when the matching retry callback fires,
- do not drop, mutate, or free the packet prematurely.

Examples:

- `src/learning_gem5/part2/simple_memobj.cc:65`
- `src/learning_gem5/part2/simple_memobj.cc:128`
- `src/learning_gem5/part2/simple_cache.cc:73`
- `src/learning_gem5/part2/simple_cache.cc:153`

### 2. Maintain explicit blocked / outstanding state

A blocking object should track whether it already has an in-flight transaction.

Typical state:

- `blocked`
- `blockedPacket`
- `needRetry`
- source port ID / waiting port ID
- any saved original packet for upgraded requests

Examples:

- `src/learning_gem5/part2/simple_memobj.hh:60`
- `src/learning_gem5/part2/simple_memobj.hh:230`
- `src/learning_gem5/part2/simple_cache.hh:66`
- `src/learning_gem5/part2/simple_cache.hh:281`

### 3. Release internal resources before sending a response upstream

Before sending a completed response, unblock the object first when appropriate.

Reason:

- sending the response may synchronously trigger new requests in the same call chain,
- if the object is still marked blocked, it may incorrectly reject the next request.

Examples:

- `src/learning_gem5/part2/simple_memobj.cc:192`
- `src/learning_gem5/part2/simple_cache.cc:255`

### 4. Model latency with events, not immediate logic

Timing-mode delay should normally be represented by scheduled events.

Common pattern:

- accept request,
- record state,
- `schedule(..., clockEdge(latency))`,
- do the real work in the event callback.

Example:

- `src/learning_gem5/part2/simple_cache.cc:209`

If the object is cycle-based, prefer deriving from `ClockedObject`.

Example:

- `src/learning_gem5/part2/simple_cache.hh:50`

### 5. Keep functional and atomic paths in mind

Even if the main purpose is timing mode, many gem5 objects still need:

- `recvFunctional()` for debug / state inspection / functional access,
- `recvAtomic()` for non-timing paths or compatibility.

Tutorial code may panic on atomic access, but production-quality code should make this a deliberate decision.

Examples:

- `src/learning_gem5/part2/simple_memobj.hh:103`
- `src/learning_gem5/part2/simple_cache.hh:110`

### 6. Propagate address ranges correctly

If the object forwards or proxies another responder, it often needs to:

- implement `getAddrRanges()`,
- forward downstream ranges upstream,
- propagate range changes with `sendRangeChange()`.

Examples:

- `src/learning_gem5/part2/simple_memobj.cc:220`
- `src/learning_gem5/part2/simple_memobj.cc:228`
- `src/learning_gem5/part2/simple_cache.cc:411`
- `src/learning_gem5/part2/simple_cache.cc:419`

### 7. Be precise about packet ownership and lifetime

When reviewing code, explicitly track:

- who currently owns the packet,
- whether a send succeeded or failed,
- whether the packet is stored for retry,
- whether a new packet was allocated as a replacement or upgrade,
- who is responsible for deleting dynamically allocated packets.

Canonical tricky example:

- `src/learning_gem5/part2/simple_cache.cc:229`
- `src/learning_gem5/part2/simple_cache.cc:324`

### 8. Retry only when the object is truly ready

Do not send retry notifications prematurely.

Common rule:

- only call `sendRetryReq()` or `sendRetryResp()` when the port and owner state can genuinely accept reissue.

Example:

- `src/learning_gem5/part2/simple_cache.cc:93`

### 9. Consider drain behavior for in-flight transactions

If the object holds outstanding packets, queued events, internal buffers, or delayed responses, review whether it needs custom drain support.

Reference:

- `src/sim/drain.hh:208`

### 10. Consider snooping / coherence for non-trivial memory objects

If the object participates in a coherent memory hierarchy, it may also need snoop-related timing methods and `isSnooping()` behavior.

See examples such as:

- `src/mem/mem_delay.hh:83`
- `src/mem/cache/base.hh:296`

## Recommended implementation workflow

When asked to implement a new timing SimObject, follow this order:

1. Read the target directory’s `SConscript` / Python SimObject definition / existing similar objects.
2. Identify whether the object should derive from `SimObject` or `ClockedObject`.
3. Define the ports and their roles.
4. Write the minimal timing callbacks.
5. Add explicit blocked / retry / outstanding state.
6. Add event scheduling for timing latency.
7. Add functional support.
8. Add atomic support if needed.
9. Add range propagation.
10. Review packet ownership carefully.
11. Add or update tests.
12. Build and run tests inside the docker environment.

## Review checklist

When reviewing a timing SimObject, check for these common bugs:

- `recvTimingReq` returns `true` but packet state is not retained correctly.
- `sendTimingReq` / `sendTimingResp` failure path drops the packet.
- retry callback exists but does not actually resend the blocked packet.
- object remains blocked after response completion.
- object sends response before clearing blocked state.
- event scheduling uses raw ticks incorrectly instead of clock-aware APIs where needed.
- address ranges are not forwarded.
- functional path is missing or inconsistent with timing path state.
- packet allocated with `new` is never freed.
- request upgrade / replacement packet loses original request semantics.
- multiple CPU-side ports are present but return path does not preserve source port.
- in-flight events or queues are ignored for drain.

## Useful canonical examples

### Simple blocking forwarder

- `src/learning_gem5/part2/simple_memobj.hh`
- `src/learning_gem5/part2/simple_memobj.cc`

### Simple timing cache with event scheduling

- `src/learning_gem5/part2/simple_cache.hh`
- `src/learning_gem5/part2/simple_cache.cc`

### Queued bridge / delay model

- `src/mem/mem_delay.hh`

### Production cache hierarchy reference

- `src/mem/cache/base.hh`

## Response style for this skill

When using this skill in an answer:

- explain timing behavior in terms of request path, response path, retry path, and event path,
- cite concrete file locations using `path:line`,
- distinguish clearly between `RequestPort` and `ResponsePort`,
- call out blocked-state and packet-lifetime issues explicitly,
- avoid hand-wavy descriptions like “just forward the packet”,
- prefer explaining the minimal correct state machine.

## If asked to generate code

Before writing code:

- inspect the existing directory structure and similar SimObjects,
- match local naming and file organization,
- update build glue only where necessary,
- keep the implementation minimal and task-focused,
- do not add unrelated abstractions.

After writing code in this repository:

- add or update relevant tests,
- run required build/test/style commands inside docker per `.agent/CLAUDE.md`.
