import { expect, test } from "@playwright/test";
import { ensureAuthenticated } from "./helpers/auth";

test("Runtime speech workflows remain opt-in and keep results local", async ({ page }) => {
  const transcriptions: Array<Record<string, unknown>> = [];
  const syntheses: Array<Record<string, unknown>> = [];

  await page.route("**/api/engine/speech/transcribe", async (route) => {
    transcriptions.push(route.request().postDataJSON() as Record<string, unknown>);
    await route.fulfill({ json: {
      schema_version: 1,
      engine: "whisper.cpp",
      model: "ggml-base.bin",
      transcript: "Local speech result.",
      audio: { format: "wav", size_bytes: 4096 },
      execution: { device: "cpu", threads: 4, language: "en" },
      privacy: {
        local_process_only: true, network_used: false, audio_returned: false,
        audio_persisted_by_axiom: false, transcript_persisted_by_axiom: false,
        model_downloaded: false,
      },
      scope_note: "Local whisper.cpp only.",
    } });
  });

  await page.route("**/api/engine/speech/synthesize", async (route) => {
    syntheses.push(route.request().postDataJSON() as Record<string, unknown>);
    await route.fulfill({ json: {
      schema_version: 1,
      engine: "piper",
      model: "en_US-local.onnx",
      audio_base64: "UklGRg==",
      audio: {
        format: "wav", size_bytes: 2048, channels: 1,
        sample_rate_hz: 22050, sample_width_bytes: 2, duration_seconds: 0.5,
      },
      execution: { device: "cpu", speaker_id: 0 },
      privacy: {
        local_process_only: true, network_used: false, text_persisted_by_axiom: false,
        audio_persisted_by_axiom: false, model_downloaded: false,
      },
      scope_note: "Local Piper only.",
    } });
  });

  await ensureAuthenticated(page);
  await page.goto("/?page=runtime");
  await expect(page.getByRole("heading", { name: "Local speech", exact: true })).toBeVisible();
  expect(transcriptions).toHaveLength(0);
  expect(syntheses).toHaveLength(0);

  await page.getByLabel("Whisper model file").fill("/models/ggml-base.bin");
  await page.getByLabel("Workspace audio path").fill("data/audio/meeting.wav");
  await page.getByLabel("Language").fill("en");
  await page.getByLabel("CPU threads").fill("4");
  await page.getByRole("button", { name: "Transcribe audio" }).click();
  await expect(page.getByText("TRANSCRIPTION COMPLETE", { exact: true })).toBeVisible();
  await expect(page.getByText("Local speech result.", { exact: true })).toBeVisible();
  expect(transcriptions).toEqual([{
    model_path: "/models/ggml-base.bin",
    audio_path: "data/audio/meeting.wav",
    language: "en",
    threads: 4,
    use_accelerator: false,
  }]);

  await page.getByLabel("Piper voice model file").fill("/models/en_US-local.onnx");
  await page.getByLabel("Text to speak").fill("Hello from local AXIOM.");
  await page.getByLabel("Speaker ID").fill("0");
  await page.getByRole("button", { name: "Synthesize speech" }).click();
  await expect(page.getByText("SPEECH GENERATED", { exact: true })).toBeVisible();
  const audioSource = await page.locator("audio[aria-label='Locally generated speech']").getAttribute("src");
  expect(audioSource).toBe("data:audio/wav;base64,UklGRg==");
  expect(syntheses).toEqual([{
    model_path: "/models/en_US-local.onnx",
    text: "Hello from local AXIOM.",
    speaker_id: 0,
    use_accelerator: false,
  }]);
});
