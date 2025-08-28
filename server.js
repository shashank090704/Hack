const express = require("express");
const bodyParser = require("body-parser");
const { spawn } = require("child_process");
const cors = require("cors");

const app = express();

// ---------------------
// Middleware
// ---------------------
app.use(cors({
  origin: "http://localhost:3000",  // your frontend (React)
  methods: ["GET", "POST"],
  credentials: true
}));

app.use(bodyParser.json({ limit: "50mb" }));

// ---------------------
// Prediction Route
// ---------------------
app.post("/predict", (req, res) => {
  const inputData = req.body; // frontend JSON

  // Spawn Python process
  const python = spawn("python", ["predict.py", "--topk", "10"]);

  let result = "";
  let errorMsg = "";

  // Send input JSON to Python
  python.stdin.write(JSON.stringify(inputData));
  python.stdin.end();

  // Collect Python stdout
  python.stdout.on("data", (data) => {
    result += data.toString();
  });

  // Collect Python stderr
  python.stderr.on("data", (data) => {
    errorMsg += data.toString();
  });

  // On process close
  python.on("close", (code) => {
    if (code !== 0) {
      return res.status(500).json({
        error: errorMsg || `Python process failed with exit code ${code}`
      });
    }

    try {
      // Parse JSON output from Python
      const parsed = JSON.parse(result);

      // Beautify response
      const response = {
        top_k: parsed.top_k,
        suspicious_users: parsed.results.map((u, idx) => ({
          rank: idx + 1,
          user_id: u.user_id,
          flags: u.flag_count,
          probability: `${(u.predicted_proba * 100).toFixed(2)}%`,
          last_known_ip: u.last_known_ip || null,
          last_online: u.last_online || null,
          current_mobile_no: u.current_mobile_no || null,
          last_device_logged: u.last_device_logged || null
        })),
      };

      res.json(response);

    } catch (err) {
      res.status(500).json({
        error: "Invalid JSON output from Python",
        raw: result
      });
    }
  });
});

// ---------------------
// Start Server
// ---------------------
const PORT = 5000;
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
});
