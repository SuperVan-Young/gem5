# Copyright (c) 2026
# All rights reserved.

import argparse
import json
from pathlib import Path


def _build_html(data):
    payload = json.dumps(data, ensure_ascii=False)
    title = "gem5 NPU Profiling Gantt"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: sans-serif;
    }}
    body {{
      margin: 0;
      background: #f5f7fb;
      color: #1f2937;
    }}
    .page {{
      padding: 16px 20px 24px;
    }}
    .header {{
      margin-bottom: 12px;
    }}
    .header h1 {{
      margin: 0 0 4px;
      font-size: 24px;
    }}
    .summary {{
      color: #4b5563;
      font-size: 14px;
    }}
    .controls {{
      display: flex;
      gap: 12px;
      align-items: center;
      margin: 12px 0 16px;
      flex-wrap: wrap;
    }}
    .controls button {{
      border: 1px solid #cbd5e1;
      background: white;
      border-radius: 8px;
      padding: 6px 12px;
      cursor: pointer;
    }}
    .controls input[type="range"] {{
      width: 240px;
    }}
    .timeline-shell {{
      border: 1px solid #cbd5e1;
      border-radius: 12px;
      background: white;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }}
    .timeline-viewport {{
      overflow: auto;
      max-height: 78vh;
    }}
    .axis-row {{
      display: grid;
      grid-template-columns: 220px 1fr;
      border-bottom: 1px solid #e5e7eb;
    }}
    .axis-row.header-row {{
      position: sticky;
      top: 0;
      z-index: 3;
      background: #eef2ff;
    }}
    .axis-label {{
      position: sticky;
      left: 0;
      z-index: 2;
      background: #f8fafc;
      border-right: 1px solid #e5e7eb;
      padding: 8px 12px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 2px;
      font-size: 13px;
    }}
    .header-row .axis-label {{
      background: #eef2ff;
      font-weight: 600;
    }}
    .axis-label .name {{
      font-weight: 600;
    }}
    .axis-label .sub {{
      color: #64748b;
      font-size: 12px;
    }}
    .track {{
      position: relative;
      min-height: 42px;
      background-image:
        linear-gradient(
          to right,
          rgba(148, 163, 184, 0.18) 1px,
          transparent 1px
        );
      background-size: var(--grid-step, 120px) 100%;
    }}
    .tick-label {{
      position: absolute;
      top: 6px;
      transform: translateX(-50%);
      color: #475569;
      font-size: 12px;
      white-space: nowrap;
    }}
    .bar {{
      position: absolute;
      top: 8px;
      height: 26px;
      border-radius: 8px;
      border: 1px solid rgba(15, 23, 42, 0.18);
      box-sizing: border-box;
      cursor: pointer;
      overflow: hidden;
      box-shadow: 0 2px 8px rgba(15, 23, 42, 0.12);
    }}
    .bar span {{
      display: block;
      padding: 4px 8px;
      font-size: 12px;
      line-height: 16px;
      color: rgba(15, 23, 42, 0.92);
      white-space: nowrap;
      text-overflow: ellipsis;
      overflow: hidden;
    }}
    .tooltip {{
      position: fixed;
      display: none;
      pointer-events: none;
      max-width: 520px;
      padding: 10px 12px;
      border-radius: 10px;
      background: rgba(15, 23, 42, 0.96);
      color: #e5eefc;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.35);
      z-index: 9999;
      font-size: 12px;
    }}
    .tooltip pre {{
      margin: 8px 0 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, monospace;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <h1>{title}</h1>
      <div id="summary" class="summary"></div>
    </div>
    <div class="controls">
      <button id="zoom-out" type="button">缩小</button>
      <button id="zoom-in" type="button">放大</button>
      <button id="fit" type="button">完整时长</button>
      <button id="focus" type="button">聚焦事件</button>
      <label>
        缩放
        <input id="zoom" type="range" min="0" max="16" step="1" value="0">
      </label>
      <span id="zoom-label"></span>
    </div>
    <div class="timeline-shell">
      <div id="viewport" class="timeline-viewport"></div>
    </div>
  </div>
  <div id="tooltip" class="tooltip"></div>
  <script>
    const profileData = {payload};
    const viewport = document.getElementById("viewport");
    const tooltip = document.getElementById("tooltip");
    const zoomInput = document.getElementById("zoom");
    const zoomLabel = document.getElementById("zoom-label");
    const summary = document.getElementById("summary");
    const labelWidth = 220;

    const events = profileData.events;
    const axes = profileData.axes;
    const minTick = Math.min(...events.map((event) => event.start_tick));
    const maxTick = Math.max(...events.map((event) => event.end_tick));
    const tickSpan = Math.max(1, maxTick - minTick);
    const lanePitch = 34;
    const minTrackHeight = 42;

    const palette = [
      ["hsl(220 78% 72%)", "hsl(220 78% 82%)"],
      ["hsl(150 55% 70%)", "hsl(150 55% 80%)"],
      ["hsl(28 90% 72%)", "hsl(28 90% 82%)"],
      ["hsl(286 62% 74%)", "hsl(286 62% 84%)"],
      ["hsl(0 75% 76%)", "hsl(0 75% 85%)"],
      ["hsl(190 70% 72%)", "hsl(190 70% 82%)"],
    ];

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }}

    function formatMaybeHex(key, value) {{
      if (typeof value !== "number") {{
        return value;
      }}
      if (
        key.includes("addr") ||
        key.includes("mask") ||
        key.includes("cfg") ||
        key.includes("word") ||
        key.includes("base")
      ) {{
        return "0x" + value.toString(16);
      }}
      return value;
    }}

    function buildTooltip(event) {{
      const detailObject = {{}};
      for (const [key, value] of Object.entries(event.details || {{}})) {{
        detailObject[key] = formatMaybeHex(key, value);
      }}
      const rawWords = (event.raw_words || []).map(
        (word, index) => `w${{index}}=0x${{Number(word).toString(16)}}`
      );
      return `
        <div><strong>${{escapeHtml(event.title)}}</strong></div>
        <div>SEU: ${{escapeHtml(event.seu_name)}}</div>
        <div>
          Tick: ${{event.start_tick}} → ${{event.end_tick}}
          (duration=${{event.duration}})
        </div>
        <div>
          Queue: ${{event.issue_queue}} |
          Kind: ${{escapeHtml(event.macro_kind)}}
        </div>
        <pre>${{escapeHtml(JSON.stringify(detailObject, null, 2))}}</pre>
        <pre>${{escapeHtml(rawWords.join("\\n"))}}</pre>
      `;
    }}

    function niceStep(span) {{
      const rough = Math.max(1, span / 8);
      const magnitude = 10 ** Math.floor(Math.log10(rough));
      const normalized = rough / magnitude;
      if (normalized <= 1) return magnitude;
      if (normalized <= 2) return 2 * magnitude;
      if (normalized <= 5) return 5 * magnitude;
      return 10 * magnitude;
    }}

    function buildAxisLayouts() {{
      const layouts = new Map();

      for (const axisInfo of axes) {{
        const axisEvents = events
          .filter((event) => event.axis === axisInfo.axis)
          .slice()
          .sort((left, right) => (
            left.start_tick - right.start_tick ||
            left.end_tick - right.end_tick ||
            left.macro_id - right.macro_id
          ));
        const laneEnds = [];

        for (const event of axisEvents) {{
          let lane = 0;
          while (
            lane < laneEnds.length &&
            event.start_tick < laneEnds[lane]
          ) {{
            lane += 1;
          }}
          event.lane = lane;
          laneEnds[lane] = event.end_tick;
        }}

        layouts.set(axisInfo.axis, {{
          events: axisEvents,
          laneCount: Math.max(1, laneEnds.length),
        }});
      }}

      return layouts;
    }}

    const axisLayouts = buildAxisLayouts();
    const boundaryTicks = Array.from(
      new Set(events.flatMap((event) => [event.start_tick, event.end_tick]))
    ).sort((left, right) => left - right);
    let minBoundaryGap = tickSpan;
    for (let index = 1; index < boundaryTicks.length; index++) {{
      const gap = boundaryTicks[index] - boundaryTicks[index - 1];
      if (gap > 0) {{
        minBoundaryGap = Math.min(minBoundaryGap, gap);
      }}
    }}
    minBoundaryGap = Math.max(1, minBoundaryGap);

    function fitScale() {{
      const viewportWidth = Math.max(
        320,
        viewport.clientWidth || window.innerWidth - 80
      );
      return Math.max(1e-12, (viewportWidth - 260) / tickSpan);
    }}

    function currentScale() {{
      return fitScale() * (2 ** Number(zoomInput.value));
    }}

    function setSmartZoom() {{
      const baseScale = fitScale();
      const desiredScale = Math.max(baseScale, 48 / minBoundaryGap);
      const zoomValue = Math.ceil(Math.log2(desiredScale / baseScale));
      zoomInput.value = String(
        Math.max(
          Number(zoomInput.min),
          Math.min(Number(zoomInput.max), zoomValue)
        )
      );
    }}

    function render() {{
      const pxPerTick = currentScale();
      const viewportWidth = Math.max(
        320,
        viewport.clientWidth || window.innerWidth - 80
      );
      const contentWidth = Math.max(
        viewportWidth - 4,
        Math.ceil(tickSpan * pxPerTick) + 32
      );
      const gridStepTick = niceStep(tickSpan);
      const gridStepPx = Math.max(80, gridStepTick * pxPerTick);
      const tickPerPx = 1 / pxPerTick;
      const tickPerPxText = tickPerPx.toLocaleString(
        undefined,
        {{ maximumFractionDigits: 2 }}
      );
      zoomLabel.textContent =
        `${{tickPerPxText}} ` +
        `tick/px`;
      summary.textContent =
        `source=${{profileData.source}} | axes=${{profileData.axis_count}} ` +
        `| events=${{profileData.event_count}} ` +
        `| tick=${{minTick}}..${{maxTick}}`;

      const parts = [];
      parts.push(`
        <div class="axis-row header-row">
          <div class="axis-label">
            <div class="name">SEU Axis</div>
            <div class="sub">tick ${{minTick}} .. ${{maxTick}}</div>
          </div>
          <div class="track"
               style="width:${{contentWidth}}px;
                      --grid-step:${{gridStepPx}}px">
            ${{
              Array.from(
                {{ length: Math.floor(tickSpan / gridStepTick) + 1 }},
                (_, index) => {{
                  const tick = minTick + index * gridStepTick;
                  const left = (tick - minTick) * pxPerTick;
                  return `
                    <div class="tick-label" style="left:${{left}}px">
                      ${{tick}}
                    </div>
                  `;
                }}
              ).join("")
            }}
          </div>
        </div>
      `);

      for (let axisIndex = 0; axisIndex < axes.length; axisIndex++) {{
        const axisInfo = axes[axisIndex];
        const axisLayout = axisLayouts.get(axisInfo.axis);
        const axisEvents = axisLayout.events;
        const trackHeight = Math.max(
          minTrackHeight,
          axisLayout.laneCount * lanePitch + 16
        );
        const [primaryColor, secondaryColor] =
          palette[axisIndex % palette.length];
        const bars = axisEvents.map((event) => {{
          const left = (event.start_tick - minTick) * pxPerTick;
          const width = Math.max(2, event.duration * pxPerTick);
          const top = 8 + (event.lane || 0) * lanePitch;
          const color =
            event.axis_order % 2 === 0 ? primaryColor : secondaryColor;
          return `
            <div class="bar"
                 style="left:${{left}}px; top:${{top}}px;
                        width:${{width}}px;
                         background:${{color}}"
                 data-tooltip="${{encodeURIComponent(buildTooltip(event))}}">
              <span>${{escapeHtml(event.title)}}</span>
            </div>
          `;
        }}).join("");

        parts.push(`
          <div class="axis-row">
            <div class="axis-label">
              <div class="name">${{escapeHtml(axisInfo.axis_display)}}</div>
              <div class="sub">${{escapeHtml(axisInfo.axis)}}</div>
            </div>
            <div class="track"
                  style="width:${{contentWidth}}px;
                         height:${{trackHeight}}px;
                         --grid-step:${{gridStepPx}}px">
              ${{bars}}
            </div>
          </div>
        `);
      }}

      viewport.innerHTML = parts.join("");

      for (const bar of viewport.querySelectorAll(".bar")) {{
        bar.addEventListener("mouseenter", (event) => {{
          tooltip.style.display = "block";
          tooltip.innerHTML = decodeURIComponent(
            event.currentTarget.dataset.tooltip
          );
        }});
        bar.addEventListener("mousemove", (event) => {{
          tooltip.style.left = `${{event.clientX + 16}}px`;
          tooltip.style.top = `${{event.clientY + 16}}px`;
        }});
        bar.addEventListener("mouseleave", () => {{
          tooltip.style.display = "none";
        }});
      }}
    }}

    document.getElementById("zoom-in").addEventListener("click", () => {{
      zoomInput.value = Math.min(
        Number(zoomInput.max),
        Number(zoomInput.value) + 1
      );
      render();
    }});
    document.getElementById("zoom-out").addEventListener("click", () => {{
      zoomInput.value = Math.max(
        Number(zoomInput.min),
        Number(zoomInput.value) - 1
      );
      render();
    }});
    document.getElementById("fit").addEventListener("click", () => {{
      zoomInput.value = 0;
      render();
    }});
    document.getElementById("focus").addEventListener("click", () => {{
      setSmartZoom();
      render();
    }});
    viewport.addEventListener(
      "wheel",
      (event) => {{
        if (!events.length) {{
          return;
        }}

        event.preventDefault();
        const oldScale = currentScale();
        const rect = viewport.getBoundingClientRect();
        const localX = event.clientX - rect.left;
        const anchorX = Math.max(labelWidth, localX);
        const trackOffset = viewport.scrollLeft + anchorX - labelWidth;
        const anchorTick = minTick + (trackOffset / oldScale);
        const direction = event.deltaY < 0 ? 1 : -1;
        const step = event.shiftKey ? 2 : 1;
        const nextValue = Math.max(
          Number(zoomInput.min),
          Math.min(
            Number(zoomInput.max),
            Number(zoomInput.value) + (direction * step)
          )
        );

        if (nextValue === Number(zoomInput.value)) {{
          return;
        }}

        zoomInput.value = String(nextValue);
        render();

        const newScale = currentScale();
        const newTrackOffset = (anchorTick - minTick) * newScale;
        viewport.scrollLeft = Math.max(
          0,
          newTrackOffset - (anchorX - labelWidth)
        );
      }},
      {{ passive: false }}
    );
    zoomInput.addEventListener("input", render);
    window.addEventListener("resize", render);

    setSmartZoom();
    render();
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="parsed profile JSON path",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="HTML output path",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    data = json.loads(input_path.read_text())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_build_html(data))


if __name__ == "__main__":
    main()
