import { ImageSegmenter, FilesetResolver } 
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

const CAMERA_URL = "http://192.168.0.150/capture"; // coloque o IP do ESP32-CAM
const BOT_TOKEN = "SEU_BOT_TOKEN";
const CHAT_ID = "SEU_CHAT_ID";

async function loadModel() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  return await ImageSegmenter.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite",
    },
    outputCategoryMask: true
  });
}

async function run() {
  const segmenter = await loadModel();

  async function process() {
    const img = document.getElementById("inputImage");
    img.src = CAMERA_URL + "?t=" + Date.now();

    img.onload = async () => {
      const canvas = document.getElementById("outputCanvas");
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d");

      const result = await segmenter.segment(img);
      const mask = result.categoryMask.getAsImageData();

      ctx.putImageData(mask, 0, 0);

      sendToTelegram(canvas);
    };

    setTimeout(process, 2000);
  }

  process();
}

async function sendToTelegram(canvas) {
  const blob = await new Promise(resolve =>
    canvas.toBlob(resolve, "image/png")
  );

  const form = new FormData();
  form.append("chat_id", CHAT_ID);
  form.append("photo", blob, "segmentada.png");

  await fetch(`https://api.telegram.org/bot${BOT_TOKEN}/sendPhoto`, {
    method: "POST",
    body: form
  });

  console.log("Enviado ao Telegram!");
}

run();
