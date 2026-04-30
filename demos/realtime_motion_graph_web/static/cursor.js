// Custom cursor: a small constellation of amber particles in loose orbit
// around the pointer, plus a precise center dot. The orbit elongates
// along the axis of a nearby HUD bar (top → wider, left/right → taller),
// so the cluster leans toward grabbable surfaces like iron filings around
// a magnet — telegraphing affordance without obstructing the click point.
//
// Implementation:
//   - N particles with spring-damper physics following slot positions
//     that rotate slowly around the pointer.
//   - Orbit radius is per-axis (rX, rY); the bar attractor stretches one
//     axis based on the pointer's proximity to a bar edge.
//   - The center dot follows the pointer via CSS variables (no physics);
//     it's the click target and must be precise.
//
// CSS handles inflate/deflate via body.cursor-idle (existing 2s timer).

const N_PARTICLES     = 6;
const ORBIT_RADIUS    = 24;     // px, baseline orbit radius
const SPRING_K        = 0.18;   // 0..1 portion of distance closed per frame
const DAMPING         = 0.78;   // 0..1 velocity retained per frame
const ROTATION_SPEED  = 0.0004; // rad / ms, slow ambient orbit
const ATTRACTOR_RANGE = 140;    // px from a bar edge where pull begins
const ATTRACTOR_GAIN  = 0.85;   // peak orbit-radius stretch (fraction)

export function initCursor() {
  const container = document.querySelector(".cursor-cluster");
  if (!container) return;

  const particles = [];
  for (let i = 0; i < N_PARTICLES; i++) {
    const el = document.createElement("div");
    el.className = "cursor-particle";
    container.appendChild(el);
    particles.push({
      el,
      x: window.innerWidth / 2,
      y: window.innerHeight / 2,
      vx: 0, vy: 0,
      slot: (i / N_PARTICLES) * Math.PI * 2,
    });
  }

  let mouseX = window.innerWidth / 2;
  let mouseY = window.innerHeight / 2;
  document.addEventListener("mousemove", (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
  }, { passive: true });

  // Squared falloff: gentle far away, strong as the cursor enters the bar.
  const proximity = (dist) => {
    if (dist <= 0) return 1;
    if (dist >= ATTRACTOR_RANGE) return 0;
    const f = 1 - dist / ATTRACTOR_RANGE;
    return f * f;
  };

  let lastT = 0;
  function loop(t) {
    const dt = lastT ? Math.min(50, t - lastT) : 16;
    lastT = t;

    document.body.style.setProperty("--cursor-x", `${mouseX}px`);
    document.body.style.setProperty("--cursor-y", `${mouseY}px`);

    // Bar attractors → per-axis orbit stretch. Top bar widens X (along
    // its scrub axis); left + right bars stretch Y. Bottom isn't a
    // slider so it's intentionally absent.
    const vw = window.innerWidth;
    const stretchX = proximity(mouseY);
    const stretchY = Math.max(proximity(mouseX), proximity(vw - mouseX));

    const rX = ORBIT_RADIUS * (1 + stretchX * ATTRACTOR_GAIN);
    const rY = ORBIT_RADIUS * (1 + stretchY * ATTRACTOR_GAIN);

    for (const p of particles) {
      p.slot += ROTATION_SPEED * dt;
      const targetX = mouseX + Math.cos(p.slot) * rX;
      const targetY = mouseY + Math.sin(p.slot) * rY;

      p.vx = (p.vx + (targetX - p.x) * SPRING_K) * DAMPING;
      p.vy = (p.vy + (targetY - p.y) * SPRING_K) * DAMPING;
      p.x += p.vx;
      p.y += p.vy;

      p.el.style.transform = `translate3d(${p.x}px, ${p.y}px, 0)`;
    }

    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}
