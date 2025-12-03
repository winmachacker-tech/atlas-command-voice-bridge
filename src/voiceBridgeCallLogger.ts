// FILE: src/voiceBridgeCallLogger.ts
//
// Purpose:
// - Tiny helper used by your voice bridge server (src/server.ts)
//   to log calls into the Supabase Edge Function sales-call-log.
// - Works for BOTH outbound + inbound calls.
// - Adds support for direction = "INBOUND" for inbound calls.
//
// Security:
// - Uses ONLY backend env vars (SUPABASE_URL, SUPABASE_ANON_KEY, ATLAS_VOICE_INTERNAL_SECRET, VOICE_SERVER_WEBHOOK_SECRET).
// - Sends a shared secret header so only your bridge can call the Edge Function.
// - Does NOT expose any keys to the browser.

export type CallDirection = "INBOUND" | "OUTBOUND";

export interface CallLogPayload {
  callSid: string;
  orgId?: string | null;
  prospectId?: string | null;
  status?: string;
  direction?: CallDirection;
  toNumber?: string | null;
  fromNumber?: string | null;
  transcript?: string | null;
  aiSummary?: string | null;
  startedAt?: string | null;
  endedAt?: string | null;
  modelUsed?: string | null;
  recordingUrl?: string | null;
  recordingDurationSeconds?: number | null;
}

function getSalesCallLogUrl(): string {
  const supabaseUrl = process.env.SUPABASE_URL;
  if (!supabaseUrl) {
    throw new Error(
      "[voiceBridgeCallLogger] SUPABASE_URL is not set in the Node environment",
    );
  }
  return `${supabaseUrl.replace(/\/$/, "")}/functions/v1/sales-call-log`;
}

function getSupabaseAnonKey(): string {
  const key = process.env.SUPABASE_ANON_KEY;
  if (!key) {
    throw new Error(
      "[voiceBridgeCallLogger] SUPABASE_ANON_KEY is not set in the Node environment",
    );
  }
  return key;
}

function getInternalSecret(): string {
  const secret =
    process.env.ATLAS_VOICE_INTERNAL_SECRET ??
    process.env.VOICE_SERVER_WEBHOOK_SECRET ??
    "";

  if (!secret) {
    throw new Error(
      "[voiceBridgeCallLogger] ATLAS_VOICE_INTERNAL_SECRET or VOICE_SERVER_WEBHOOK_SECRET is not set in the Node environment",
    );
  }
  return secret;
}

export async function logCallToSupabase(
  payload: CallLogPayload,
): Promise<void> {
  const {
    callSid,
    orgId,
    prospectId,
    status,
    direction,
    toNumber,
    fromNumber,
    transcript,
    aiSummary,
    startedAt,
    endedAt,
    modelUsed,
    recordingUrl,
    recordingDurationSeconds,
  } = payload;

  if (!callSid) {
    throw new Error("[voiceBridgeCallLogger] callSid is required");
  }

  const url = getSalesCallLogUrl();
  const anonKey = getSupabaseAnonKey();
  const secret = getInternalSecret();

  const body: any = {
    twilio_call_sid: callSid,
    org_id: orgId ?? null,
    prospect_id: prospectId ?? null,
    status: status ?? "COMPLETED",
    direction: direction ?? "OUTBOUND",
    to_number: toNumber ?? null,
    from_number: fromNumber ?? null,
    transcript: transcript ?? null,
    ai_summary: aiSummary ?? null,
    started_at: startedAt ?? null,
    ended_at: endedAt ?? null,
    model: modelUsed ?? null,
    recording_url: recordingUrl ?? null,
    recording_duration_seconds: recordingDurationSeconds ?? null,
  };

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${anonKey}`,
        "x-atlas-voice-secret": secret,
      },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const text = await res.text().catch(() => "<no body>");
      console.error(
        "[voiceBridgeCallLogger] sales-call-log returned non-OK:",
        res.status,
        text,
      );
    } else {
      const json = await res.json().catch(() => ({}));
      console.log(
        "[voiceBridgeCallLogger] sales-call-log OK:",
        res.status,
        json,
      );
    }
  } catch (err) {
    console.error("[voiceBridgeCallLogger] Error calling sales-call-log:", err);
  }
}