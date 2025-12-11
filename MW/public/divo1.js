const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { createCanvas, Image } = require("canvas");

const {
  FilesetResolver,
  PoseLandmarker
} = require("@mediapipe/tasks-vision");

const app = express();
app.use(express.json());

const storage = multer.diskStorage({
  destination: "./uploads",
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  }
});
const upload = multer({ storage });

async function loadImage(pathFile) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = pathFile;
  });
}

let poseLandmarker;

async function carregarModelo() {
  const vision = await FilesetResolver.forVisionTasks(
    "node_modules/@mediapipe/tasks-vision/wasm"
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "node_modules/@mediapipe/tasks-vision/models/pose_landmarker_full.task"
    },
    runningMode: "IMAGE",

    outputSegmentationMasks: true
  });

  console.log("Modelo Mediapipe carregado!");
}

carregarModelo();


app.post("/processar", upload.single("foto"), async (req, res) => {
  try {
    if (!poseLandmarker) {
      return res.status(503).json({ erro: "Modelo não carregado ainda" });
    }

    const caminho = req.file.path;
    const img = await loadImage(caminho);

    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);

    const mpImage = {
      width: img.width,
      height: img.height,
      data: ctx.getImageData(0, 0, img.width, img.height).data
    };


    const result = poseLandmarker.detect(mpImage);

    if (!result || !result.landmarks || result.landmarks.length === 0) {
      return res.json({ status: "nenhuma pessoa detectada" });
    }

    const pontos = result.landmarks[0];

    const nose = pontos[0];
    const leftShoulder = pontos[11];
    const rightShoulder = pontos[12];

    const mask = result.segmentationMasks[0];

    res.json({
      status: "ok",
      nose,
      leftShoulder,
      rightShoulder,
      segmentationMask: mask
    });

  } catch (e) {
    console.error(e);
    res.status(500).json({ erro: e.message });
  }
});

app.listen(3000, () => {
  console.log("Servidor rodando em http://localhost:3000");
});
