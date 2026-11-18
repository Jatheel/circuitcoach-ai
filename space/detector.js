// CircuitCoach AI - detection helpers (no browser APIs here, so they can be unit-tested).

const CLASS_NAMES = ["1-5-Volt-Battery", "3-3-Volt-Battery", "7-Segment-Display", "9-Volt-Battery",
  "Arduino-Mega", "Arduino-Nano", "Arduino-Uno", "BJT-Transistor", "Bluetooth-Module", "Breadboard",
  "Bridge-Rectifier", "Buck-Converter", "Buzzer", "Capacitor-10mf", "Capacitor-470mf", "DC-Motor",
  "Diode", "ESP32", "ESP32-CAM", "FT-232-USB-Serial-Module", "Film-Capacitor", "Fuse", "Fuse-Base",
  "GSM-Module", "Gas-Sensor", "Heat-Sink", "High-Voltage-Ceramic-Capacitor", "Humidity-Sensor",
  "IC-Base-14-Pin", "IC-Base-28-Pin", "IC-Chip", "IGBT", "IR-Sensor", "Inductor", "Keypad",
  "LCD-Display", "LDR-Sensor", "LED-Light", "Low-Voltage-Ceramic-Capacitor", "MLC-Capacitor", "MOSFET",
  "Motion-Sensor", "Motor-Driver", "NTC-Thermistor", "OLED-Display", "Pin-Header", "Push-Switch",
  "RFID-Scanner", "Raindrops-Module", "Relay-Module", "Resistor", "Rocker-Switch", "Servo-Motor",
  "Soil-Moisture-Sensor", "Sonar-Sensor", "TCRT5000", "Tact-Switch", "Taper-Potentiometer",
  "Trimmer-Potentiometer", "Water-Sensor", "Zener-Diode"];

const INPUT = 640;

// Letterbox geometry: scale the image to fit 640x640, centred, grey padding.
function letterbox(w, h, size = INPUT) {
  const scale = Math.min(size / w, size / h);
  const nw = Math.round(w * scale), nh = Math.round(h * scale);
  return { scale, nw, nh, padX: Math.floor((size - nw) / 2), padY: Math.floor((size - nh) / 2) };
}

// RGBA pixels of the 640x640 letterboxed canvas -> Float32 CHW in [0, 1].
function toCHW(rgba, size = INPUT) {
  const n = size * size, out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    out[i] = rgba[4 * i] / 255;
    out[n + i] = rgba[4 * i + 1] / 255;
    out[2 * n + i] = rgba[4 * i + 2] / 255;
  }
  return out;
}

function iou(a, b) {
  const x0 = Math.max(a.x1, b.x1), y0 = Math.max(a.y1, b.y1);
  const x1 = Math.min(a.x2, b.x2), y1 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  const area = (r) => (r.x2 - r.x1) * (r.y2 - r.y1);
  return inter / (area(a) + area(b) - inter + 1e-9);
}

function nms(boxes, iouThr = 0.7, maxDet = 300) {
  boxes.sort((a, b) => b.conf - a.conf);
  const keep = [];
  for (const b of boxes) {
    if (keep.length >= maxDet) break;
    if (keep.every((k) => k.cls !== b.cls || iou(k, b) <= iouThr)) keep.push(b);
  }
  return keep;
}

// Model output -> boxes in ORIGINAL image pixels.
// Handles (1, N, 6) [x1,y1,x2,y2,conf,cls] (NMS embedded) and raw (1, 4+nc, A) [cx,cy,w,h,scores...].
function postprocess(data, dims, lb, imgW, imgH, minConf = 0.1) {
  const boxes = [];
  const unmap = (x1, y1, x2, y2) => ({
    x1: Math.min(Math.max((x1 - lb.padX) / lb.scale, 0), imgW),
    y1: Math.min(Math.max((y1 - lb.padY) / lb.scale, 0), imgH),
    x2: Math.min(Math.max((x2 - lb.padX) / lb.scale, 0), imgW),
    y2: Math.min(Math.max((y2 - lb.padY) / lb.scale, 0), imgH),
  });
  if (dims.length === 3 && dims[2] === 6) {
    for (let i = 0; i < dims[1]; i++) {
      const r = data.subarray(6 * i, 6 * i + 6);
      if (r[4] < minConf) continue;
      boxes.push({ ...unmap(r[0], r[1], r[2], r[3]), conf: r[4], cls: Math.round(r[5]) });
    }
    return boxes.sort((a, b) => b.conf - a.conf);
  }
  if (dims.length === 3 && dims[1] > 4) {
    const nc = dims[1] - 4, A = dims[2];
    for (let a = 0; a < A; a++) {
      let best = -1, conf = 0;
      for (let c = 0; c < nc; c++) {
        const s = data[(4 + c) * A + a];
        if (s > conf) { conf = s; best = c; }
      }
      if (conf < minConf) continue;
      const cx = data[a], cy = data[A + a], w = data[2 * A + a], h = data[3 * A + a];
      boxes.push({ ...unmap(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), conf, cls: best });
    }
    return nms(boxes);
  }
  throw new Error("Unexpected model output shape: " + JSON.stringify(dims));
}

// Group detections above the threshold by class: [{name, count, best}], best first.
function summarise(boxes, conf) {
  const g = new Map();
  for (const b of boxes) {
    if (b.conf < conf) continue;
    const name = CLASS_NAMES[b.cls] || `class ${b.cls}`;
    const cur = g.get(name) || { name, count: 0, best: 0 };
    cur.count += 1; cur.best = Math.max(cur.best, b.conf);
    g.set(name, cur);
  }
  return [...g.values()].sort((a, b) => b.best - a.best);
}

if (typeof module !== "undefined") {
  module.exports = { CLASS_NAMES, letterbox, toCHW, iou, nms, postprocess, summarise };
}
