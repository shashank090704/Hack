// const express = require("express");
// const bodyParser = require("body-parser");
// const { spawn } = require("child_process");
// const fs = require("fs");

// const app = express();

// // Increase JSON limit to handle large files
// app.use(bodyParser.json({ limit: "50mb" }));  
// app.use(bodyParser.urlencoded({ limit: "50mb", extended: true }));

// app.post("/predict", (req, res) => {
//   const inputData = req.body;

//   // Save request JSON into Testing_User.json
//   fs.writeFileSync("Testing_User.json", JSON.stringify(inputData, null, 2));

//   // Run Python script
//   const python = spawn("python", ["predict.py", "--file", "Testing_User.json", "--topk", "10"]);

//   let result = "";

//   python.stdout.on("data", (data) => {
//     result += data.toString();
//   });

//   python.stderr.on("data", (data) => {
//     console.error(`stderr: ${data}`);
//   });

//   python.on("close", (code) => {
//     console.log(`child process exited with code ${code}`);
//     res.json({ output: result });
//   });
// });

// app.listen(5000, () => {
//   console.log("Server running on http://localhost:5000");
// });
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////\
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// const express = require("express");
// const bodyParser = require("body-parser");
// const { spawn } = require("child_process");

// const app = express();

// // Increase payload limit
// app.use(bodyParser.json({ limit: "50mb" }));

// app.post("/predict", (req, res) => {
//   const inputData = req.body; // frontend JSON

//   // Spawn Python process
//   const python = spawn("python", ["predict.py", "--topk", "10"]);

//   let result = "";
//   let errorMsg = "";

//   // Send JSON directly to Python stdin
//   python.stdin.write(JSON.stringify(inputData));
//   python.stdin.end();

//   python.stdout.on("data", (data) => {
//     result += data.toString();
//   });

//   python.stderr.on("data", (data) => {
//     errorMsg += data.toString();
//   });

//   python.on("close", (code) => {
//     if (code !== 0) {
//       return res.status(500).json({ error: errorMsg });
//     }
//     res.json({ output: result.trim() });
//   });
// });

// app.listen(5000, () => {
//   console.log("🚀 Server running on http://localhost:5000");
// });

const express = require("express");
const bodyParser = require("body-parser");
const { spawn } = require("child_process");
import cors from "cors";
const app = express();

app.use(cors({
  origin: ["http://localhost:3000"], // allow frontend
  methods: ["GET", "POST"], // allowed methods
  credentials: true
}));
app.use(bodyParser.json({ limit: "50mb" }));

app.post("/predict", (req, res) => {
  const inputData = req.body; // frontend JSON

  const python = spawn("python", ["predict.py", "--topk", "10"]);

  let result = "";
  let errorMsg = "";

  // Send input JSON to Python
  python.stdin.write(JSON.stringify(inputData));
  python.stdin.end();

  python.stdout.on("data", (data) => {
    result += data.toString();
  });

  python.stderr.on("data", (data) => {
    errorMsg += data.toString();
  });

  python.on("close", (code) => {
    if (code !== 0) {
      return res.status(500).json({ error: errorMsg || "Python process failed" });
    }

    try {
      // Parse string safely to JSON
      const parsed = JSON.parse(result);

      // Beautify response
      const response = {
        top_k: parsed.top_k,
        suspicious_users: parsed.results.map((u, idx) => ({
          rank: idx + 1,
          user_id: u.user_id,
          flags: u.flag_count,
          probability: `${(u.predicted_proba * 100).toFixed(2)}%`,
        })),
      };

      res.json(response);
    } catch (err) {
      res.status(500).json({ error: "Invalid JSON output from Python", raw: result });
    }
  });
});

app.listen(5000, () => {
  console.log("🚀 Server running on http://localhost:5000");
});
