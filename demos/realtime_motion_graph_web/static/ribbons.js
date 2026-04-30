// Multi-color organic ribbons painted along the three slider edges.
// Replaces the original 2 px amber level meter with bands of color
// inspired by the artist's relief paintings — multiple bright bezier
// strokes of paint flowing together with black outlines between them.
//
// Each bar's SVG fills its edge zone; ribbon paths are allowed to bleed
// out of the viewBox into the video gutter (overflow: visible). Slider
// fill drives the path length from the anchor end; time animates a
// 2-octave trig-noise field across the band; kick punches the writhe
// amplitude (and the existing --bloom-amount widens stroke widths via
// CSS, keeping visual lockstep with the rest of the kick-driven UI).
//
// SVG coordinate system per bar:
//   horizontal bars (top): viewBox 0..ALONG x 0..ACROSS, anchor at x=0
//   vertical bars (L/R):   viewBox 0..ACROSS x 0..ALONG, anchor at y=ALONG (bottom)
// flipAlong handles the bottom-anchor for the vertical bars.

// Four-color band — picked from the artwork as the most contrasty
// distinct hues so a thin set of strokes still reads as multi-color.
const PALETTE = [
  "#ff2d8d", // hot pink
  "#2858d4", // electric blue
  "#6ed548", // lime green
  "#ffd633", // yellow
];

const SVG_NS = "http://www.w3.org/2000/svg";

const ALONG = 1000;             // viewBox length axis
const ACROSS = 100;             // viewBox cross axis (= bar thickness, hud-thickness)
const SEGMENTS = 36;            // polyline resolution
const RIBBON_SPACING = 3;       // viewBox units between adjacent ribbon centers (tight band)
const NOISE_AMP_BASE = 6;       // baseline writhe amplitude (low-key)
const NOISE_AMP_KICK = 8;       // additional writhe amplitude per unit kick
const INWARD_DISTANCE = 8;      // band offset from the inner edge of the bar zone

// SVG x=0 is the LEFT edge of the bar zone. For the top/left bars the
// inner edge (touching the video) is at across=ACROSS; for the right
// bar the inner edge is at across=0. innerSign captures that so we
// always sit a fixed INWARD_DISTANCE from the video boundary.
const BAR_CONFIG = [
  { sel: ".install-edge-top",   horizontal: true,  flipAlong: false, innerSign: +1 },
  { sel: ".install-edge-left",  horizontal: false, flipAlong: true,  innerSign: +1 },
  { sel: ".install-edge-right", horizontal: false, flipAlong: true,  innerSign: -1 },
];

function makeSvg(horizontal) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", "install-ribbons");
  svg.setAttribute(
    "viewBox",
    horizontal ? `0 0 ${ALONG} ${ACROSS}` : `0 0 ${ACROSS} ${ALONG}`,
  );
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("aria-hidden", "true");
  return svg;
}

function makePath(stroke, klass) {
  const p = document.createElementNS(SVG_NS, "path");
  p.setAttribute("fill", "none");
  p.setAttribute("stroke", stroke);
  p.setAttribute("stroke-linecap", "round");
  p.setAttribute("stroke-linejoin", "round");
  p.setAttribute("class", klass);
  return p;
}

export function initRibbons() {
  const bars = [];
  for (const cfg of BAR_CONFIG) {
    const edge = document.querySelector(cfg.sel);
    if (!edge) continue;

    // Drop the legacy 2 px amber bar; the SVG owns the meter now.
    const oldBar = edge.querySelector(".install-edge-bar");
    if (oldBar) oldBar.remove();

    const svg = makeSvg(cfg.horizontal);
    const ribbons = [];
    // One stroke per ribbon — no black outline. The stroke's natural
    // anti-aliased edge is what we want; a hard outline reads as
    // attention-grabbing relief, which fights the video.
    for (let i = 0; i < PALETTE.length; i++) {
      const path = makePath(PALETTE[i], "ribbon-color");
      svg.appendChild(path);
      ribbons.push(path);
    }
    edge.appendChild(svg);
    bars.push({
      edge,
      ribbons,
      horizontal: cfg.horizontal,
      flipAlong: cfg.flipAlong,
      innerSign: cfg.innerSign,
    });
  }
  return bars;
}

function ribbonPathD(progress, ribbonIdx, time, kick, bar) {
  if (progress < 0.001) return "";
  const drawLen = progress * ALONG;
  const lateral = (ribbonIdx - (PALETTE.length - 1) / 2) * RIBBON_SPACING;
  const phase = ribbonIdx * 0.8;
  const writheAmp = NOISE_AMP_BASE + kick * NOISE_AMP_KICK;
  // Anchor the band a fixed distance from the inner edge so it sits
  // where the original 2 px amber line did — close to the video, not
  // floating in the middle of the bar zone.
  const center = bar.innerSign > 0
    ? ACROSS - INWARD_DISTANCE
    : INWARD_DISTANCE;

  let d = "";
  let prevX = 0, prevY = 0, lastX = 0, lastY = 0;
  for (let i = 0; i <= SEGMENTS; i++) {
    const t = i / SEGMENTS;
    const along = t * drawLen;
    // Two octaves of out-of-phase trig — each ribbon gets a different
    // phase so they writhe out of sync, matching the artwork's flow.
    const noise =
      Math.sin(along * 0.012 + time * 1.3 + phase) * 0.7 +
      Math.sin(along * 0.025 - time * 0.9 + phase * 1.4) * 0.3;
    const across = center + lateral + noise * writheAmp;

    let x, y;
    if (bar.horizontal) {
      x = along;
      y = across;
    } else {
      x = across;
      y = bar.flipAlong ? ALONG - along : along;
    }
    if (i > 0) { prevX = lastX; prevY = lastY; }
    d += (i === 0 ? "M" : "L") + x.toFixed(1) + " " + y.toFixed(1) + " ";
    lastX = x; lastY = y;
  }

  // Curving arrowhead at the leading edge: a 180° curl perpendicular
  // to the path's last heading. Even ribbons curl one way, odd the
  // other, so the band's tips fan out instead of all hooking the same
  // direction. innerSign factors in so "even = inward" is consistent
  // across bars (the right bar's perpendicular is mirrored from the
  // left bar's). Curl radius pulses a touch on kick.
  if (drawLen > 8) {
    const dx = lastX - prevX;
    const dy = lastY - prevY;
    const segLen = Math.hypot(dx, dy) || 1;
    const ux = dx / segLen;
    const uy = dy / segLen;
    const altSign = (ribbonIdx % 2 === 0) ? 1 : -1;
    const sign = altSign * (bar.innerSign > 0 ? 1 : -1);
    const px = -uy * sign;
    const py = ux * sign;
    const r = 7 + kick * 3;
    const cx = lastX + px * r;
    const cy = lastY + py * r;
    const curlSteps = 8;
    for (let j = 1; j <= curlSteps; j++) {
      const a = (j / curlSteps) * Math.PI;
      const sa = Math.sin(a);
      const ca = Math.cos(a);
      // Rotate the (-px, -py) vector around (cx, cy) toward the
      // forward direction (ux, uy) over 0..π.
      const rx = -px * ca + ux * sa;
      const ry = -py * ca + uy * sa;
      d += "L" + (cx + rx * r).toFixed(1) + " " + (cy + ry * r).toFixed(1) + " ";
    }
  }
  return d;
}

export function tickRibbons(bars, time, kick) {
  for (const bar of bars) {
    const fill = parseFloat(bar.edge.style.getPropertyValue("--fill")) || 0;
    for (let i = 0; i < bar.ribbons.length; i++) {
      bar.ribbons[i].setAttribute("d", ribbonPathD(fill, i, time, kick, bar));
    }
  }
}
