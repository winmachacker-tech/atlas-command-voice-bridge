// FILE: src/server.ts
// Purpose:
// - Bridge between Twilio Programmable Voice Media Streams and OpenAI Realtime.
// - Handle full duplex audio:
//     • Twilio -> (µ-law 8kHz) -> decode & upsample -> OpenAI Realtime (pcm16 16kHz)
//     • OpenAI -> (g711_ulaw 8kHz) base64 -> Twilio media payload
//
// - PLUS call intelligence:
//     • Enable Whisper transcription of caller audio
//     • Capture Dipsy (assistant) text
//     • Build a full transcript in memory
//     • On Twilio "stop":
//          - Generate an AI summary from that transcript (chat completion)
//          - Send transcript + summary to Supabase via the sales-call-log Edge Function
//
// Audio path:
//   Twilio (G.711 µ-law, 8kHz)  → this bridge → OpenAI Realtime (pcm16 16kHz)
//   OpenAI (g711_ulaw 8kHz)     → this bridge → Twilio (as media payload)
//
//
// Transcription path:
//   - session.update enables `input_audio_transcription` with model "whisper-1"
//   - We listen for:
//       • conversation.item.input_audio_transcription.completed  (Caller text)
//       • response.output_text.delta + response.completed        (Dipsy text)
//   - We append to ctx.transcript as:
//       Caller: ...\n
//       Dipsy:  ...\n
//
// Supabase logging (via Edge Function):
//   - On Twilio "start", we read `callSid` and store on ctx
//   - On Twilio "stop", we:
//       1) Generate an AI summary from the in-memory transcript (if long enough)
//       2) Call Supabase Edge Function sales-call-log with:
//            { twilio_call_sid, transcript, ai_summary, direction, ... }
//       3) The Edge Function (running with service role) updates public.sales_calls
//
// Env vars (in this voice-bridge project):
//   OPENAI_API_KEY
//   OPENAI_REALTIME_MODEL       (optional, default gpt-4o-realtime-preview-2024-12-17)
//   OPENAI_VOICE_STYLE          (optional, default shimmer)
//   OPENAI_SUMMARY_MODEL        (optional, default gpt-4.1-mini)
//
//
//   SUPABASE_URL                (used by voiceBridgeCallLogger to hit sales-call-log)
//   ATLAS_VOICE_INTERNAL_SECRET (shared secret checked by sales-call-log)
//   VOICE_SERVER_WEBHOOK_SECRET (optional fallback shared secret)
//
//   PORT                        (optional, default 8080)
//   BRIDGE_VERSION              (optional, string to tag logs/health output)
//
//
// Security:
// - This bridge NEVER exposes secrets to the browser.
// - Database writes happen inside the Supabase Edge Function using service_role.
// - This server only knows the Edge Function URL + shared secret via env.

import http from "http";
import express from "express";
import WebSocket, { WebSocketServer } from "ws";
import path from "path";
import fs from "fs";
import dotenv from "dotenv";
import { randomUUID } from "crypto";
import { logCallToSupabase } from "./voiceBridgeCallLogger";
import type { CallDirection } from "./voiceBridgeCallLogger";

// ---------- Explicit .env load ----------

const envPath = path.resolve(__dirname, "../.env");
dotenv.config({ path: envPath });
console.log("[Config] Loaded .env from:", envPath);

// ---------- Config ----------

const PORT = parseInt(process.env.PORT || "8080", 10);

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  console.error("[VoiceBridge] Missing OPENAI_API_KEY in environment.");
  process.exit(1);
}

const OPENAI_REALTIME_MODEL =
  process.env.OPENAI_REALTIME_MODEL || "gpt-4o-realtime-preview-2024-12-17";
const OPENAI_VOICE_STYLE = process.env.OPENAI_VOICE_STYLE || "shimmer";

// Model used for AI summaries of the transcript (normal chat completions).
const OPENAI_SUMMARY_MODEL =
  process.env.OPENAI_SUMMARY_MODEL || "gpt-4.1-mini";

// Optional version tag for logs/health
const BRIDGE_VERSION = process.env.BRIDGE_VERSION || "v1.6.1";

console.log("[Config] OPENAI_REALTIME_MODEL:", OPENAI_REALTIME_MODEL);
console.log("[Config] OPENAI_VOICE_STYLE:", OPENAI_VOICE_STYLE);
console.log("[Config] OPENAI_SUMMARY_MODEL:", OPENAI_SUMMARY_MODEL);

// ---------- Load Dipsy V2 summary prompt from file ----------

const SUMMARY_PROMPT_PATH = path.resolve(
  __dirname,
  "../prompts/dipsy-call-summary-v2.txt",
);

let SUMMARY_SYSTEM_PROMPT = "";
try {
  SUMMARY_SYSTEM_PROMPT = fs.readFileSync(SUMMARY_PROMPT_PATH, "utf8");
  console.log(
    "[Config] Loaded Dipsy V2 summary prompt from:",
    SUMMARY_PROMPT_PATH,
  );
} catch (err) {
  console.error(
    "[Config] FAILED to load Dipsy V2 summary prompt from:",
    SUMMARY_PROMPT_PATH,
    "Error:",
    (err as any)?.message ?? String(err),
  );
  // Fallback: keep empty; generateSummaryFromTranscript will handle this case.
}

// ---------- Load Dipsy V2 CALL prompt from file ----------

const CALL_PROMPT_PATH = path.resolve(
  __dirname,
  "../prompts/dipsy-call-agent-v2.txt",
);

let CALL_SYSTEM_PROMPT = "";
try {
  CALL_SYSTEM_PROMPT = fs.readFileSync(CALL_PROMPT_PATH, "utf8");
  console.log(
    "[Config] Loaded Dipsy V2 call prompt from:",
    CALL_PROMPT_PATH,
  );
} catch (err) {
  console.error(
    "[Config] FAILED to load Dipsy V2 call prompt from:",
    CALL_PROMPT_PATH,
    "Error:",
    (err as any)?.message ?? String(err),
  );
  // Fallback: we will use a minimal built-in prompt if this file is missing.
}

// ---------- Structured logging helper ----------

function logStructured(event: string, fields: Record<string, any> = {}) {
  const payload = {
    ts: new Date().toISOString(),
    service: "atlas-voice-bridge",
    version: BRIDGE_VERSION,
    event,
    ...fields,
  };
  console.log(JSON.stringify(payload));
}

// ---------- Helpers: µ-law <-> PCM ----------

// Convert one 8-bit µ-law sample to 16-bit linear PCM.
function mulawToLinearSample(uVal: number): number {
  uVal = ~uVal & 0xff;
  const sign = uVal & 0x80;
  let exponent = (uVal >> 4) & 0x07;
  let mantissa = uVal & 0x0f;
  let sample = ((mantissa << 3) + 0x84) << exponent;
  sample -= 0x84;
  return sign ? -sample : sample;
}

// Decode an entire Buffer of µ-law bytes -> PCM16 (16-bit signed LE).
function decodeMulawToPcm16(mulawBuf: Buffer): Buffer {
  const samples = new Int16Array(mulawBuf.length);
  for (let i = 0; i < mulawBuf.length; i++) {
    samples[i] = mulawToLinearSample(mulawBuf[i]);
  }
  return Buffer.from(samples.buffer);
}

// Very simple upsample from 8kHz -> 16kHz by duplicating each sample.
function upsample8kTo16k(pcm8k: Buffer): Buffer {
  const inputSamples = pcm8k.length / 2; // 2 bytes per sample
  const outSamples = inputSamples * 2;
  const out = Buffer.alloc(outSamples * 2);
  for (let i = 0; i < inputSamples; i++) {
    const sample = pcm8k.readInt16LE(i * 2);
    // Write the same sample twice
    out.writeInt16LE(sample, 2 * (2 * i));
    out.writeInt16LE(sample, 2 * (2 * i + 1));
  }
  return out;
}

// ---------- Types for internal state ----------

type CallType = "FIRST" | "FOLLOWUP";

interface TwilioStreamContext {
  // static per WebSocket connection
  connectionId: string;
  // correlation across systems: prefers callSid, then streamSid, then connectionId
  correlationId: string | null;

  streamSid: string | null;
  callSid: string | null;
  twilioWs: WebSocket;
  openaiWs: WebSocket | null;
  openaiReady: boolean;

  // transcript state
  transcript: string;
  assistantBuffer: string;

  // VAD / turn-taking state
  humanSpeaking: boolean; // our best guess if the caller is currently speaking
  lastHumanSpeechAt: number | null; // last time we detected caller voice (ms since epoch)

  // call metadata
  direction: CallDirection; // "INBOUND" or "OUTBOUND"
  callType: CallType; // "FIRST" or "FOLLOWUP"
  lastSummary: string | null;
  lastTranscript: string | null;
}

// ---------- Express + HTTP + WS setup ----------

const app = express();

// Simple root response for sanity.
app.get("/", (_req, res) => {
  res.send("Atlas Command Voice Bridge is running.");
});

// Health endpoint used by Supabase sales-voice-health function.
// This MUST match VOICE_BRIDGE_HEALTH_URL, e.g.:
//   https://<ngrok-subdomain>.ngrok-free.dev/health
app.get("/health", (_req, res) => {
  res.json({
    ok: true,
    service: "atlas-voice-bridge",
    version: BRIDGE_VERSION,
    uptime_seconds: process.uptime(),
    timestamp: new Date().toISOString(),
  });
});

const server = http.createServer(app);

// WebSocket server JUST for Twilio media streams.
// NOTE: Path must match the TwiML <Stream url="wss://.../twilio"> value.
const wss = new WebSocketServer({ server, path: "/twilio" });

wss.on("connection", (twilioWs: WebSocket) => {
  const connectionId = randomUUID();

  logStructured("twilio_ws_connection_open", {
    connection_id: connectionId,
  });
  console.log("\n[Twilio] New Media Stream WebSocket connection.");

  const ctx: TwilioStreamContext = {
    connectionId,
    correlationId: null,
    streamSid: null,
    callSid: null,
    twilioWs,
    openaiWs: null,
    openaiReady: false,
    transcript: "",
    assistantBuffer: "",
    humanSpeaking: false,
    lastHumanSpeechAt: null,
    direction: "OUTBOUND", // default; overridden by Twilio customParameters
    callType: "FIRST",
    lastSummary: null,
    lastTranscript: null,
  };

  twilioWs.on("message", async (msg: WebSocket.RawData) => {
    try {
      const text = msg.toString();
      const data = JSON.parse(text);
      const event = data.event;

      switch (event) {
        case "start":
          await handleTwilioStart(ctx, data);
          break;
        case "media":
          await handleTwilioMedia(ctx, data);
          break;
        case "mark":
          logStructured("twilio_mark_event", {
            connection_id: ctx.connectionId,
            correlation_id: ctx.correlationId,
            mark_name: data.mark?.name,
          });
          console.log("[Twilio] mark:", data.mark?.name);
          break;
        case "stop":
          logStructured("twilio_stop_event", {
            connection_id: ctx.connectionId,
            correlation_id: ctx.correlationId,
            call_sid: ctx.callSid,
            stream_sid: ctx.streamSid,
          });
          console.log(
            "[Twilio] stop event received. Saving transcript & summary via sales-call-log, then closing sockets.",
          );
          await saveTranscriptAndSummary(ctx);
          cleanup(ctx);
          break;
        default:
          logStructured("twilio_unknown_event", {
            connection_id: ctx.connectionId,
            raw_event: event,
          });
          console.log("[Twilio] Unknown event:", event);
      }
    } catch (err: any) {
      logStructured("twilio_message_handler_error", {
        connection_id: ctx.connectionId,
        correlation_id: ctx.correlationId,
        error: err?.message ?? String(err),
      });
      console.error("[Twilio] Error handling message:", err);
    }
  });

  twilioWs.on("close", () => {
    logStructured("twilio_ws_connection_closed", {
      connection_id: ctx.connectionId,
      correlation_id: ctx.correlationId,
      call_sid: ctx.callSid,
      stream_sid: ctx.streamSid,
    });
    console.log("[Twilio] WebSocket closed.");
    cleanup(ctx);
  });

  twilioWs.on("error", (err) => {
    logStructured("twilio_ws_connection_error", {
      connection_id: ctx.connectionId,
      correlation_id: ctx.correlationId,
      call_sid: ctx.callSid,
      stream_sid: ctx.streamSid,
      error: (err as any)?.message ?? String(err),
    });
    console.error("[Twilio] WebSocket error:", err);
    cleanup(ctx);
  });
});

// ---------- Handlers ----------

async function handleTwilioStart(
  ctx: TwilioStreamContext,
  data: any,
): Promise<void> {
  const streamSid = data.start?.streamSid;
  const callSid = data.start?.callSid; // Twilio includes callSid in start event
  ctx.streamSid = streamSid || null;
  ctx.callSid = callSid || null;

  // Detect inbound vs outbound + follow-up metadata from Twilio customParameters
  const params = data.start?.customParameters ?? {};

  if (params.direction === "INBOUND") {
    ctx.direction = "INBOUND";
    console.log("[Direction] INBOUND call detected.");
  } else {
    ctx.direction = "OUTBOUND";
    console.log("[Direction] OUTBOUND call detected.");
  }

  const rawCallType = (params.call_type as string | undefined) || "FIRST";
  ctx.callType = rawCallType === "FOLLOWUP" ? "FOLLOWUP" : "FIRST";

  ctx.lastSummary =
    typeof params.last_summary === "string" && params.last_summary.length > 0
      ? params.last_summary
      : null;

  ctx.lastTranscript =
    typeof params.last_transcript === "string" &&
    params.last_transcript.length > 0
      ? params.last_transcript
      : null;

  // Choose a correlationId that is stable across logs + Supabase:
  // Prefer callSid, then streamSid, then connectionId.
  ctx.correlationId = callSid || streamSid || ctx.connectionId;

  logStructured("twilio_start_event", {
    connection_id: ctx.connectionId,
    correlation_id: ctx.correlationId,
    call_sid: callSid,
    stream_sid: streamSid,
    direction: ctx.direction,
    call_type: ctx.callType,
    has_last_summary: !!ctx.lastSummary,
    has_last_transcript: !!ctx.lastTranscript,
  });

  console.log("[Twilio] start event. streamSid =", streamSid);
  console.log("[Twilio] callSid =", callSid);

  // Open connection to OpenAI Realtime for this call
  const url = `wss://api.openai.com/v1/realtime?model=${encodeURIComponent(
    OPENAI_REALTIME_MODEL,
  )}`;

  logStructured("openai_realtime_connecting", {
    connection_id: ctx.connectionId,
    correlation_id: ctx.correlationId,
    model: OPENAI_REALTIME_MODEL,
    url,
  });

  const openaiWs = new WebSocket(url, {
    headers: {
      Authorization: `Bearer ${OPENAI_API_KEY}`,
      "OpenAI-Beta": "realtime=v1",
    },
  });

  ctx.openaiWs = openaiWs;

  openaiWs.on("open", () => {
    logStructured("openai_realtime_connected", {
      connection_id: ctx.connectionId,
      correlation_id: ctx.correlationId,
    });
    console.log("[OpenAI] Realtime WebSocket connected.");
    ctx.openaiReady = true;

    // ----------------- Build Dipsy instructions with FOLLOW-UP memory -----------------

    // Use external Dipsy V2 CALL prompt if available; otherwise a short fallback.
    const baseInstructions =
      CALL_SYSTEM_PROMPT && CALL_SYSTEM_PROMPT.trim().length > 0
        ? CALL_SYSTEM_PROMPT
        : `
You are **Dipsy**, the AI voice sales assistant for Atlas Command, an AI-powered TMS and operations platform built specifically for trucking carriers.

Your job:
- Have natural, human-sounding sales conversations with carrier owners/dispatchers.
- Discover if Atlas Command is a good fit.
- Clearly explain value in their language.
- Guide them to an appropriate next step (demo, follow-up, or email) without being pushy.

You are talking to real people on live phone calls. Your responses are converted to speech.

==================================================
CORE PERSONALITY & TONE
==================================================

You are:
- Confident, warm, calm, and helpful.
- Rational and slightly tactical, not salesy or bubbly.
- Very knowledgeable about trucking operations, dispatch, and TMS workflows.
- Respectful of people’s time. You never ramble.

Tone rules:
- Default mode: **professional + friendly**.
- Speak in short, clear sentences, especially early in the call.
- Avoid jargon unless they use it first.
- Never sound annoyed, defensive, or sarcastic.
- You can use light, professional humor only if the person is clearly relaxed and open.

Important constraints:
- **Never ask for MC or DOT numbers.** Atlas already has that info. You may reference “your operation” or “your fleet” instead.
- **Mobile app truth:** If asked about a mobile app, you MUST say:
  - “We don’t have a standalone mobile app yet, but Atlas is fully mobile-responsive in the browser, so you can use it on any phone or tablet. A native mobile app is planned for around Q1 of 2026.”

==================================================
CONVERSATION STAGES (STATE MACHINE)
==================================================

Internally, follow this structure:

1. GREETING
2. PERMISSION TO TALK (TIME CHECK)
3. CONTEXT & PURPOSE
4. DISCOVERY & QUALIFICATION
5. VALUE PITCH (TAILORED, ROI-FOCUSED)
6. OBJECTION HANDLING (LOOP AS NEEDED)
7. CLOSE ATTEMPT (mini → soft → calendar)
8. EMAIL CAPTURE FALLBACK (if no calendar)
9. WRAP UP & FOLLOW-UP COMMITMENT

You can move back and forth between stages fluidly, but DO NOT skip DISCOVERY & QUALIFICATION unless they explicitly ask you to “just explain what this is quickly”.

==================================================
ADAPTIVE CONVERSATION ENGINE
==================================================

You must adapt your behavior in real time based on their mood, tone, and responses.

Approximate mood using text cues:

- RUSHED:
  - Short, clipped answers (“yeah”, “nope”, “real quick”).
  - Phrases like “I’ve only got a minute”, “I’m driving”, “make it quick”.
- RELAXED / OPEN:
  - Longer answers, small talk, mild jokes, or follow-up questions.
- SKEPTICAL:
  - Phrases like “we’re fine”, “we’re all set”, “we already have something”.
- IN PAIN / FRUSTRATED:
  - Complaining about communication, missed loads, chaos, spreadsheets, poor TMS, etc.

Mode switching rules:

- If they sound RUSHED:
  - Switch to **direct + professional**.
  - Say things like: “Got it, I’ll keep this very short.”
  - Ask at most 1–2 key questions, then go to a quick value statement and a simple next step.
- If they’re RELAXED / OPEN:
  - Stay **friendly + professional**.
  - Ask several discovery questions and explore pain/ROI in more detail.
- If they’re SKEPTICAL:
  - Stay calm and rational.
  - Acknowledge their current setup and ask: “What would you need to see to consider changing what you’re using now?”
- If they’re clearly IN PAIN:
  - Show empathy and focus on relief and ROI.
  - Ask how this pain shows up day to day and what it costs them in time, money, or stress.

Pacing:
- Start with medium-length responses.
- If they seem rushed or give very short answers, shorten your responses.
- If they ask detailed questions, you can expand but still keep structure and clarity.

==================================================
GREETING & PURPOSE
==================================================

1. GREETING & IDENTITY
   - Example:
     - “Hey, this is Dipsy calling with Atlas Command. I’ll be quick — I’m an AI assistant helping carriers see if our tools are a fit.”

   - Always disclose you’re AI early, but smoothly:
     - “I’m an AI voice assistant on the team, and my job is just to see if this is worth a deeper look for you.”

2. PERMISSION TO TALK
   - Always check time:
     - “Is now a bad time, or do you have about a minute?”

   If they say:
   - “I’m busy / not a good time”:
     - Use the “I’m busy” objection handling (see below).
   - “Yeah, go ahead”:
     - Move to brief context and then discovery.

3. CONTEXT & HIGH-LEVEL PURPOSE
   - Very short statement of purpose:
     - “We built Atlas Command as an AI-powered TMS and ops assistant specifically for trucking carriers. I just want to ask a couple quick questions and see if it could actually help your operation.”

==================================================
DISCOVERY & QUALIFICATION LOGIC
==================================================

Your priority after greeting is to understand their operation and pain.

Ask **2–5 natural questions**, adjusting for their time and mood. You don’t have to ask every question; prioritize the ones that fit the flow.

Key topics:
1. Fleet & team (without asking MC/DOT):
   - “Roughly how many trucks are you running right now?”
   - “Do you have people dedicated to dispatch, or are you wearing all the hats yourself?”

2. Current tools / TMS:
   - “What are you using to manage loads and dispatch today — a TMS, spreadsheets, something else?”

3. Volume / complexity:
   - “About how many loads are you moving in a typical week?”
   - “Do you deal mostly with simple one-pick, one-drop loads, or a lot of multi-stop trips?”

4. Pain & frustration:
   - “What’s the most annoying part of how you’re managing things right now?”
   - “Where do you feel the most chaos — is it tracking drivers, paperwork, or just keeping up with everything?”

5. Timeline / urgency:
   - “Are you actively looking to improve this now, or more like thinking ahead for the next few months?”

As they answer, you should:
- Reflect back their words briefly (“So you’re running about 12 trucks and most of the chaos is keeping track of drivers while you’re on the road yourself?”).
- Ask one focused follow-up that deepens understanding of pain or process.

==================================================
CARRIER INTELLIGENCE & LEAD SCORING (A/B/C)
==================================================

Internally, assess lead quality based on what you hear:

A-LEAD (High value, priority):
- Roughly 10+ trucks OR multiple dispatchers/teams.
- Clear pain (they complain about current tools, chaos, or lost time).
- Near-term timeline (“we’re looking”, “we need to fix this”, “soon”).

B-LEAD (Good fit, growing):
- Around 3–10 trucks or a clearly growing operation.
- Some pain, open to change, but not urgent.
- Might be using a basic TMS or spreadsheets.

C-LEAD (Light-touch, nurture):
- 1–2 trucks and happy with current setup.
- Low urgency, more curious than serious.
- Limited need for a full TMS right now.

Adjust your behavior:
- A-LEADS: Make sure they understand value and aim for a **calendar demo**.
- B-LEADS: Be consultative and aim for **soft close + calendar or strong email follow-up**.
- C-LEADS: Do not push hard. Focus on **email capture** and light nurturing.

==================================================
VALUE PITCH & ROI-DRIVEN LANGUAGE
==================================================

When you pitch, ALWAYS tie it back to their situation.

Examples of value framing:
- Time savings:
  - “For carriers like yours, dispatchers usually get back 1–3 hours a day because Atlas handles a lot of the busywork and communication.”
- Fewer mistakes / better visibility:
  - “You get one place to see loads, drivers, and communication, instead of juggling five different things.”
- Communication clarity:
  - “Drivers and dispatch get cleaner updates, so fewer ‘where’s my truck?’ calls.”

Rules:
- Use specific, believable outcomes (hours saved, fewer calls, fewer headaches).
- Avoid unrealistic promises (never guarantee exact revenue or specific lanes).
- Use phrases like “carriers typically see…” or “for most carriers we work with…” rather than absolute guarantees.

==================================================
OBJECTION HANDLING FRAMEWORK
==================================================

When you hear an objection, do three things:
1. Acknowledge it.
2. Briefly address it (short or long version).
3. Guide back to an appropriate close (mini, soft, calendar, or email).

Use the following playbook.

--------------------------------------------------
1) OBJECTION: “I’m busy” / “Not a good time”
--------------------------------------------------

SHORT RESPONSE (for very rushed leads):
- “Totally get it. I’ll keep this under 30 seconds.”
- “If it’s easier, we can also set a quick time later or I can just send a short email.”

LONGER RESPONSE (if they seem open but busy):
- “No problem — I know you’re juggling a lot. In one sentence: Atlas helps carriers automate a lot of dispatch and tracking so you get your time back.”
- “If I ask you just one quick question about how you’re handling dispatch today, I can tell you whether a demo is even worth your time.”

PATH BACK TO CLOSE:
- MINI CLOSE:
  - “Quick question: Are you using a TMS today, or mostly spreadsheets and texts?”
- If they still seem rushed after that:
  - Go to **email capture fallback**:
    - “What’s the best email to send you a 1-minute overview you can look at later?”

--------------------------------------------------
2) OBJECTION: “We already have a TMS”
--------------------------------------------------

SHORT RESPONSE:
- “That makes sense. A lot of the carriers we talk to are already on something.”
- “My goal isn’t to replace anything blindly — just to see if there’s a meaningful upgrade for you.”

LONGER RESPONSE:
- “Totally fair. Most carriers we work with came from another TMS — the main reasons they switch are usually time savings and better visibility.”
- “Without naming names, people often say their current system is either clunky, hard for drivers to use, or doesn’t handle their real day-to-day chaos.”
- “If you’re open to it, I’d love to know: What do you like about what you’re using now, and what you wish it did better?”

PATH BACK TO CLOSE:
- Ask a discovery question about their current TMS pain.
- If they share any frustration:
  - Use that to propose a **soft close**:
    - “Based on that, it might be worth a 15–20 minute demo just to see if Atlas handles those headaches better. Would you be open to that if we keep it focused on [their pain]?”
- If they’re genuinely happy and closed off:
  - Respect it, and shift to **email fallback**:
    - “Totally fine. If you don’t mind, I’ll send a quick email so if things change later, you’ve got an option.”

--------------------------------------------------
3) OBJECTION: “Just send me an email”
--------------------------------------------------

SHORT RESPONSE:
- “Absolutely, I can do that.”
- “To make it actually useful, let me ask one quick question so the email is tailored to you.”

LONGER RESPONSE:
- “Happy to. The last thing you need is another generic sales email.”
- “If I know how you’re handling dispatch and how many trucks you’re running, I can send a short overview that actually fits your situation.”

PATH BACK TO CLOSE:
- Ask 1–2 quick qualification questions.
- Then:
  - “Okay, based on that I’ll send you a short summary and a couple of specific ways Atlas might help.”
  - For A/B leads, add a **soft close** option:
    - “If you see something interesting in that email, would you be open to a quick walkthrough sometime this week or next?”

--------------------------------------------------
4) OBJECTION: “We’re too small”
--------------------------------------------------

SHORT RESPONSE:
- “I hear that a lot from smaller fleets.”
- “Atlas actually started with smaller carriers in mind so they don’t drown in admin as they grow.”

LONGER RESPONSE:
- “That makes sense. A lot of 1–5 truck fleets feel like they’re ‘too small’ for a system — until the paperwork, phone calls, and texts start eating their whole day.”
- “The idea with Atlas is to give you something that’s light enough for a small operation, but strong enough that you don’t have to rip everything out when you hit 10, 20, or more trucks.”

PATH BACK TO CLOSE:
- For truly small and not-urgent leads (C-leads):
  - “How about this: I’ll send you a short email with how other small fleets use Atlas as they grow. That way, when you hit your next growth spurt, you’re not starting from zero.”
- For small but clearly growing or in pain (B-leads):
  - “Since you’re already feeling some of that chaos, it might be worth a quick demo just to see if we can take a few things off your plate. Would you be open to a short walkthrough, even if you decide to wait later?”

--------------------------------------------------
5) OBJECTION: “We don’t use dispatchers”
--------------------------------------------------

SHORT RESPONSE:
- “Got it, so you’re probably handling most of this yourself.”
- “Atlas can still help even if you don’t have a dedicated dispatcher.”

LONGER RESPONSE:
- “That makes sense. A lot of smaller owners are doing everything — booking loads, talking to brokers, updating drivers, and handling paperwork.”
- “In those cases, Atlas is less about managing a big dispatch team and more about keeping everything organized so you’re not losing hours to little tasks.”

PATH BACK TO CLOSE:
- Ask:
  - “What part of that tends to eat the most of your time — is it tracking loads, communicating with drivers, or just keeping everything straight?”
- If there’s pain:
  - Suggest **soft close**:
    - “If we could show you a way to cut some of that time down, would a quick demo be worth it, even if you’re not ready to add staff yet?”
- If they truly have no pain and are content:
  - Respect it, and move to **email fallback**:
    - “Totally fair. Let me send you a short overview so if things change down the road, you’ve got an option.”

==================================================
CLOSING LOGIC & FOLLOW-UP
==================================================

You have four levels of close:

1) MINI CLOSE (low friction)
   - Used early to confirm interest.
   - Examples:
     - “From what you’ve told me, does this sound like something that could help, or not really?”
     - “On a scale of 1–10, how worth-it would it be to get some of that time back in your week?”

   - If interest is 7–10 or clearly positive → move to soft/calendar close.
   - If weak interest → go lighter and aim for email.

2) SOFT CLOSE
   - Goal: get permission to explore further, not a hard commitment.
   - Examples:
     - “If we kept it focused on [their specific pain] and made it 15–20 minutes, would you be open to seeing a quick demo?”
     - “Would you be opposed to taking a look at how Atlas could handle that part for you?”

3) CALENDAR CLOSE (for A and strong B leads)
   - Only after some discovery and value explanation.
   - Examples:
     - “It sounds like this really could help. The best next step is a short demo with the team that built Atlas, where they walk through your exact use case.”
     - “What does your schedule look like the next few days — mornings or afternoons better for you?”

   - If they verbally agree:
     - Confirm day/time window conversationally.
     - Clarify they’ll receive a calendar invite and any needed details via email.

4) EMAIL CAPTURE FALLBACK
   - Used when:
     - They’re too busy for a call.
     - They resist scheduling.
     - They only want email.
     - They’re a C-lead with low urgency.

   - Examples:
     - “What’s the best email to send you a quick overview and a couple of examples of how carriers like you are using Atlas?”
     - “I’ll keep it short and specific to the number of trucks you’re running.”

   - After capturing email:
     - Confirm it slowly, spelling back if needed.
     - Example:
       - “Just to confirm, that’s j.smith at truckingco dot com, right?”

==================================================
TRANSCRIPTION-FRIENDLY BEHAVIOR
==================================================

Because your words are transcribed, you must:

- Use clear, simple sentences.
- Avoid long, run-on speech.
- When saying email addresses:
  - Say them slowly and confirm spelling.
- Avoid lists of numbers unless necessary.
- Repeat key details (like dates or times) clearly.

==================================================
DEAD-ENDS, TRANSFER, AND STOPPING
==================================================

DEAD-ENDS:
- If someone is clearly not interested, annoyed, or repeatedly says no:
  - Respect it.
  - Example:
    - “Totally understand. I appreciate you taking the time. If things change down the road and you want to look at options, Atlas will be here. Have a good day.”
  - Then end the call.

TRANSFER TO HUMAN (if supported by the system):
- If they ask for “a real person”, “a rep”, or “someone on your team”:
  - Example:
    - “Absolutely. I can connect you with a human on the Atlas team who can go deeper.”
  - Then follow the system’s transfer behavior (the platform will handle the actual transfer).

WHEN TO STOP:
- End the call after:
  - A clear “no” and you’ve acknowledged it.
  - You’ve scheduled a demo and confirmed the email.
  - You’ve captured email and they indicate they need to go.
- Always end with a polite close:
  - “Thanks for your time today. Safe travels and have a great rest of your day.”

==================================================
HARD RULES & SAFETY
==================================================

- Never ask for MC or DOT numbers.
- Never promise guaranteed revenue, lanes, or specific rates.
- Do not give legal, medical, or financial advice.
- Stay in the lane of trucking operations, dispatch, and TMS value.
- If someone asks an unrelated or inappropriate question, gently redirect back to Atlas or end politely.

==================================================
OVERALL STYLE SUMMARY
==================================================

- You are calm, warm, and highly competent.
- You mirror their style lightly:
  - More direct with rushed/skeptical people.
  - Slightly more conversational with relaxed people.
- You always try to:
  - Understand their situation,
  - Identify pain,
  - Tie Atlas to that pain with clear value,
  - Guide them to a reasonable next step (demo or email),
  - And leave the relationship in a good place, even if they say no.
`.trim();

    const followupContext =
      ctx.callType === "FOLLOWUP"
        ? `
FOLLOW-UP CONTEXT FOR THIS SPECIFIC CALL

This is a FOLLOW-UP call with the same carrier you spoke with previously.

Here is an AI summary of the last conversation (do NOT read this verbatim, use it as your own notes):

${ctx.lastSummary ?? "(No prior AI summary was provided.)"}

If helpful, here is a short snippet from the previous transcript (again, don't read it out loud, treat it like your memory):

${ctx.lastTranscript ?? "(No prior transcript snippet was provided.)"}

Use this context to:
- Avoid repeating the same basic qualification questions.
- Briefly acknowledge the previous conversation in your own words ("Last time we talked you mentioned...").
- Check in on whether anything has changed since that call.
- Drive the conversation toward a clear next step (demo, concrete follow-up, or a clean "not a fit" if appropriate).
`.trim()
        : `
FOLLOW-UP CONTEXT FOR THIS SPECIFIC CALL

There is no previous memory loaded for this call. Treat it like a first-time conversation with this carrier and follow the normal outbound or inbound flow.
`.trim();

    const fullInstructions = `${baseInstructions.trim()}\n\n${followupContext}`.trim();

    const sessionUpdate = {
      type: "session.update",
      session: {
        instructions: fullInstructions,
        voice: OPENAI_VOICE_STYLE,
        input_audio_format: "pcm16",
        output_audio_format: "g711_ulaw",
        input_audio_transcription: {
          model: "whisper-1",
        },
        turn_detection: {
          type: "server_vad",
          threshold: 0.5,
          silence_duration_ms: 300,
          prefix_padding_ms: 300,
        },
        modalities: ["audio", "text"],
      },
    };

    openaiWs.send(JSON.stringify(sessionUpdate));
    console.log(
      "[OpenAI] Sent session.update with shimmer voice and Dipsy V2 instructions + follow-up context.",
    );
    logStructured("openai_session_update_sent", {
      connection_id: ctx.connectionId,
      correlation_id: ctx.correlationId,
      voice: OPENAI_VOICE_STYLE,
      call_type: ctx.callType,
    });

    // 🔁 Short directive for initial turn, so we don't override the whole brain again

    const inboundIntro =
      ctx.callType === "FOLLOWUP"
        ? "Start this inbound call as a brief follow-up: greet them as Dipsy with Atlas Command, acknowledge you spoke before about their operations, ask one simple follow-up question, then stop and wait for their reply."
        : "Start this inbound call by greeting the caller as Dipsy with Atlas Command, briefly mentioning you're an AI assistant helping with dispatch and operations questions, ask one simple question about what they need help with today, then stop and wait for their reply.";

    const outboundIntro =
      ctx.callType === "FOLLOWUP"
        ? "Start this outbound call as a follow-up: greet them as Dipsy with Atlas Command, remind them you spoke recently about their dispatch or TMS setup, ask one clear follow-up question to move things forward, then stop and wait for their reply."
        : "Start this outbound call by greeting them as Dipsy, an AI assistant calling on behalf of Atlas Command, and ask one simple question to confirm you're speaking with the person who handles dispatch or operations. Then stop and wait for their reply.";

    const responseCreate = {
      type: "response.create",
      response: {
        instructions: ctx.direction === "INBOUND" ? inboundIntro : outboundIntro,
      },
    };

    openaiWs.send(JSON.stringify(responseCreate));
    console.log(
      `[OpenAI] Sent initial response.create for ${ctx.direction} call, type=${ctx.callType}.`,
    );
    logStructured("openai_initial_response_create_sent", {
      connection_id: ctx.connectionId,
      correlation_id: ctx.correlationId,
      direction: ctx.direction,
      call_type: ctx.callType,
    });
  });

  openaiWs.on("message", (msg: WebSocket.RawData) => {
    try {
      const text = msg.toString();
      const event = JSON.parse(text);
      const type = event.type;

      // --- Use OpenAI's own VAD events as an additional signal ---
      if (type === "input_audio_buffer.speech_started") {
        ctx.humanSpeaking = true;
        ctx.lastHumanSpeechAt = Date.now();
        logStructured("openai_vad_speech_started", {
          connection_id: ctx.connectionId,
          correlation_id: ctx.correlationId,
          call_sid: ctx.callSid,
        });
        return;
      } else if (type === "input_audio_buffer.speech_stopped") {
        ctx.humanSpeaking = false;
        logStructured("openai_vad_speech_stopped", {
          connection_id: ctx.connectionId,
          correlation_id: ctx.correlationId,
          call_sid: ctx.callSid,
        });
        return;
      }

      if (type === "response.audio.delta") {
        // OUTBOUND AUDIO (Dipsy -> caller)
        const deltaBase64: string | undefined = event.delta;
        if (!deltaBase64) return;

        // If we think the human is speaking, DROP this audio chunk
        // so Dipsy doesn't talk over them.
        if (ctx.humanSpeaking) {
          logStructured("bridge_dropped_outbound_audio_human_speaking", {
            connection_id: ctx.connectionId,
            correlation_id: ctx.correlationId,
            call_sid: ctx.callSid,
            stream_sid: ctx.streamSid,
          });
          return;
        }

        if (ctx.twilioWs.readyState === WebSocket.OPEN && ctx.streamSid) {
          const outboundMediaMsg = {
            event: "media",
            streamSid: ctx.streamSid,
            media: {
              // Twilio expects base64-encoded 8kHz µ-law bytes.
              payload: deltaBase64,
            },
          };

          ctx.twilioWs.send(JSON.stringify(outboundMediaMsg));
          logStructured("bridge_outbound_audio_to_twilio", {
            connection_id: ctx.connectionId,
            correlation_id: ctx.correlationId,
            call_sid: ctx.callSid,
            stream_sid: ctx.streamSid,
          });
          console.log(
            "[Bridge] Sent audio delta back to Twilio (media event).",
          );
        }
      } else if (type === "response.output_text.delta") {
        // Text version of Dipsy's speech (chunked)
        const delta: string | undefined = event.delta;
        if (delta) {
          ctx.assistantBuffer += delta;
        }
      } else if (type === "response.completed") {
        // Flush Dipsy text buffer into transcript
        if (ctx.assistantBuffer.trim().length > 0) {
          ctx.transcript += `\nDipsy: ${ctx.assistantBuffer.trim()}\n`;
          ctx.assistantBuffer = "";
        }
        logStructured("openai_response_completed", {
          connection_id: ctx.connectionId,
          correlation_id: ctx.correlationId,
          call_sid: ctx.callSid,
        });
        console.log("[OpenAI] response.completed");
      } else if (
        type === "conversation.item.input_audio_transcription.completed"
      ) {
        // Whisper transcription of caller audio
        const callerText: string | undefined = event.transcript;
        if (callerText && callerText.trim().length > 0) {
          ctx.transcript += `\nCaller: ${callerText.trim()}\n`;
          logStructured("openai_caller_transcript", {
            connection_id: ctx.connectionId,
            correlation_id: ctx.correlationId,
            call_sid: ctx.callSid,
            text_length: callerText.trim().length,
          });
          console.log("[OpenAI] Caller transcript:", callerText.trim());
        }
      } else if (type === "error") {
        logStructured("openai_error_event", {
          connection_id: ctx.connectionId,
          correlation_id: ctx.correlationId,
          call_sid: ctx.callSid,
          event,
        });
        console.error("[OpenAI] error event:", event);
      } else {
        // Other events ignored for now.
      }
    } catch (err: any) {
      logStructured("openai_message_handler_error", {
        connection_id: ctx.connectionId,
        correlation_id: ctx.correlationId,
        call_sid: ctx.callSid,
        error: err?.message ?? String(err),
      });
      console.error("[OpenAI] Error handling message:", err);
    }
  });

  openaiWs.on("close", (code, reason) => {
    logStructured("openai_ws_closed", {
      connection_id: ctx.connectionId,
      correlation_id: ctx.correlationId,
      call_sid: ctx.callSid,
      code,
      reason: reason.toString(),
    });
    console.log(
      `[OpenAI] WebSocket closed. code=${code}, reason=${reason.toString()}`,
    );
    ctx.openaiReady = false;
    ctx.openaiWs = null;
  });

  openaiWs.on("error", (err) => {
    logStructured("openai_ws_error", {
      connection_id: ctx.connectionId,
      correlation_id: ctx.correlationId,
      call_sid: ctx.callSid,
      error: (err as any)?.message ?? String(err),
    });
    console.error("[OpenAI] WebSocket error:", err);
    ctx.openaiReady = false;
  });
}

async function handleTwilioMedia(
  ctx: TwilioStreamContext,
  data: any,
): Promise<void> {
  const payloadBase64: string | undefined = data.media?.payload;
  if (!payloadBase64) return;
  if (!ctx.openaiWs || !ctx.openaiReady) return;

  try {
    // 1) Decode base64 -> Buffer of µ-law bytes
    const mulawBuf = Buffer.from(payloadBase64, "base64");

    // 2) Decode µ-law -> PCM16 8kHz
    const pcm8k = decodeMulawToPcm16(mulawBuf);

    // --- Simple energy-based VAD on the Twilio side ---
    let sumAbs = 0;
    const sampleCount = pcm8k.length / 2;
    for (let i = 0; i < sampleCount; i++) {
      const sample = pcm8k.readInt16LE(i * 2);
      sumAbs += Math.abs(sample);
    }
    const avgAbs = sampleCount > 0 ? sumAbs / sampleCount : 0;
    const now = Date.now();

    const ENERGY_THRESHOLD = 500; // tweakable

    if (avgAbs > ENERGY_THRESHOLD) {
      ctx.humanSpeaking = true;
      ctx.lastHumanSpeechAt = now;
    } else {
      if (
        ctx.humanSpeaking &&
        ctx.lastHumanSpeechAt !== null &&
        now - ctx.lastHumanSpeechAt > 600
      ) {
        ctx.humanSpeaking = false;
      }
    }

    // 3) Upsample 8kHz -> 16kHz PCM16 (for OpenAI)
    const pcm16k = upsample8kTo16k(pcm8k);

    // 4) Convert to base64 for OpenAI
    const pcm16Base64 = pcm16k.toString("base64");

    // 5) Send to OpenAI Realtime as input_audio_buffer.append
    const audioAppend = {
      type: "input_audio_buffer.append",
      audio: pcm16Base64,
    };

    ctx.openaiWs.send(JSON.stringify(audioAppend));
  } catch (err: any) {
    logStructured("bridge_twilio_media_error", {
      connection_id: ctx.connectionId,
      correlation_id: ctx.correlationId,
      call_sid: ctx.callSid,
      error: err?.message ?? String(err),
    });
    console.error("[Bridge] Error handling Twilio media:", err);
  }
}

// ---------- Transcript + summary + logging via Edge Function ----------

async function saveTranscriptAndSummary(ctx: TwilioStreamContext) {
  try {
    const trimmed = ctx.transcript.trim();
    if (!ctx.callSid) {
      logStructured("calllog_missing_call_sid", {
        connection_id: ctx.connectionId,
        correlation_id: ctx.correlationId,
      });
      console.warn("[CallLog] No callSid on context; skipping transcript save.");
      return;
    }

    if (!trimmed) {
      logStructured("calllog_empty_transcript", {
        connection_id: ctx.connectionId,
        correlation_id: ctx.correlationId,
        call_sid: ctx.callSid,
      });
      console.warn(
        "[CallLog] Transcript is empty for callSid",
        ctx.callSid,
        "; skipping summary + log.",
      );
      return;
    }

    logStructured("calllog_prepare_transcript", {
      connection_id: ctx.connectionId,
      correlation_id: ctx.correlationId,
      call_sid: ctx.callSid,
      transcript_length: trimmed.length,
      direction: ctx.direction,
    });

    console.log(
      "[CallLog] Preparing transcript for callSid",
      ctx.callSid,
      "length",
      trimmed.length,
      "direction",
      ctx.direction,
    );

    let summary: string | null = null;

    if (trimmed.length >= 40) {
      summary = await generateSummaryFromTranscript(trimmed);
    } else {
      logStructured("calllog_transcript_too_short_for_summary", {
        connection_id: ctx.connectionId,
        correlation_id: ctx.correlationId,
        call_sid: ctx.callSid,
        transcript_length: trimmed.length,
      });
      console.log(
        "[CallLog] Transcript too short (< 40 chars); skipping AI summary.",
      );
    }

    // NEW: send to Supabase via helper, with direction
    await logCallToSupabase({
      callSid: ctx.callSid,
      direction: ctx.direction,
      transcript: trimmed,
      aiSummary: summary,
      orgId: process.env.ATLAS_DEFAULT_ORG_ID ?? null,
      startedAt: null,
      endedAt: new Date().toISOString(),
      toNumber: null,
      fromNumber: null,
      modelUsed: OPENAI_SUMMARY_MODEL,
      recordingUrl: null,
      recordingDurationSeconds: null,
    });
  } catch (err: any) {
    logStructured("calllog_unexpected_error", {
      connection_id: ctx.connectionId,
      correlation_id: ctx.correlationId,
      call_sid: ctx.callSid,
      error: err?.message ?? String(err),
    });
    console.error(
      "[CallLog] Unexpected error while saving transcript/summary:",
      err,
    );
  }
}

// Use OpenAI Chat Completions API to summarize the transcript.
async function generateSummaryFromTranscript(
  transcript: string,
): Promise<string | null> {
  try {
    logStructured("summary_request_start", {
      model: OPENAI_SUMMARY_MODEL,
      transcript_length: transcript.length,
    });
    console.log(
      "[Summary] Generating AI summary via model",
      OPENAI_SUMMARY_MODEL,
    );

    // Use external Dipsy V2 summary prompt if available; otherwise minimal fallback.
    const systemPrompt =
      SUMMARY_SYSTEM_PROMPT ||
      `
You are an AI assistant summarizing a sales call. The call is between "Dipsy" (an AI voice agent for Atlas Command) and a trucking carrier. Produce a concise summary and basic structured notes about interest and next steps.
`.trim();

    const userPrompt = `
Here is the full call transcript between Dipsy (the AI agent) and the prospect:

${transcript}
`.trim();

    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: OPENAI_SUMMARY_MODEL,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ],
        max_tokens: 800,
        temperature: 0.4,
      }),
    });

    if (!response.ok) {
      const text = await response.text();
      logStructured("summary_response_error", {
        status: response.status,
        body: text,
      });
      console.error(
        "[Summary] OpenAI chat completion failed. status=",
        response.status,
        "body=",
        text,
      );
      return null;
    }

    const json: any = await response.json();
    const content: string | undefined =
      json?.choices?.[0]?.message?.content ?? undefined;

    if (!content || !content.trim()) {
      logStructured("summary_empty_content", {});
      console.warn("[Summary] OpenAI response had no content.");
      return null;
    }

    logStructured("summary_response_ok", {
      summary_length: content.trim().length,
    });
    console.log("[Summary] AI summary generated.");
    return content.trim();
  } catch (err: any) {
    logStructured("summary_unexpected_error", {
      error: err?.message ?? String(err),
    });
    console.error("[Summary] Error calling OpenAI for summary:", err);
    return null;
  }
}

// ---------- Cleanup ----------

function cleanup(ctx: TwilioStreamContext) {
  if (ctx.openaiWs && ctx.openaiWs.readyState === WebSocket.OPEN) {
    ctx.openaiWs.close();
  }
  ctx.openaiWs = null;
  ctx.openaiReady = false;

  if (ctx.twilioWs && ctx.twilioWs.readyState === WebSocket.OPEN) {
    ctx.twilioWs.close();
  }
}

// ---------- Start server ----------

server.listen(PORT, () => {
  logStructured("voice_bridge_server_listening", {
    port: PORT,
  });
  console.log(`[VoiceBridge] Server listening on port ${PORT}`);
});
