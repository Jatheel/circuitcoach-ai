// Unit tests for space/detector.js  —  run with:  node tests/detector.test.js
const assert = require("assert");
const d = require("../space/detector.js");

assert.strictEqual(d.CLASS_NAMES.length, 61);

// 1280x960 photo -> scaled by 0.5 to 640x480, 80 px grey bars top and bottom
const lb = d.letterbox(1280, 960);
assert.deepStrictEqual([lb.nw, lb.nh, lb.padX, lb.padY], [640, 480, 0, 80]);

// End-to-end output (1, 300, 6): boxes are mapped back to original pixels, low scores dropped
const e2e = new Float32Array(300 * 6);
e2e.set([100, 180, 300, 280, 0.9, 6], 0);
e2e.set([0, 80, 50, 120, 0.05, 1], 6);
let out = d.postprocess(e2e, [1, 300, 6], lb, 1280, 960);
assert.strictEqual(out.length, 1);
assert.deepStrictEqual([out[0].x1, out[0].y1, out[0].x2, out[0].y2, out[0].cls], [200, 200, 600, 400, 6]);

// Raw output (1, 4+nc, anchors): decoding + per-class NMS removes the duplicate box
const nc = 61, A = 3, raw = new Float32Array((4 + nc) * A);
const set = (a, cx, cy, w, h, cls, s) => {
  raw[a] = cx; raw[A + a] = cy; raw[2 * A + a] = w; raw[3 * A + a] = h; raw[(4 + cls) * A + a] = s;
};
set(0, 200, 230, 200, 100, 6, 0.9);
set(1, 202, 231, 200, 100, 6, 0.6);
set(2, 400, 300, 50, 50, 50, 0.5);
out = d.postprocess(raw, [1, 4 + nc, A], lb, 1280, 960);
assert.deepStrictEqual(out.map((b) => d.CLASS_NAMES[b.cls]), ["Arduino-Uno", "Resistor"]);

// Grouping respects the confidence threshold
const groups = d.summarise([...out, { cls: 6, conf: 0.3 }], 0.4);
assert.deepStrictEqual(groups.map((g) => [g.name, g.count]), [["Arduino-Uno", 1], ["Resistor", 1]]);

// RGBA -> CHW float in [0, 1]
const chw = d.toCHW(new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255]), 2);
assert.deepStrictEqual(Array.from(chw), [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]);

console.log("All detector tests passed");
